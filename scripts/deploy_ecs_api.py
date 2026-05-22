import argparse
import json
import os
import sys
from typing import Any


def _parse_set_env(values: list[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for raw in values:
        # Support both:
        # 1) repeated flags: --set-env A=1 --set-env B=2
        # 2) comma-joined single flag: --set-env "A=1,B=2"
        parts = [raw]
        if "," in raw and raw.count("=") > 1:
            parts = [part.strip() for part in raw.split(",") if part.strip()]

        for value in parts:
            if "=" not in value:
                raise ValueError(f"Invalid --set-env value '{value}'. Expected KEY=VALUE.")
            key, val = value.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"Invalid --set-env value '{value}'. Empty key.")
            output[key] = val

    qdrant_host = output.get("QDRANT_HOST", "")
    if "," in qdrant_host:
        raise ValueError(
            "Invalid QDRANT_HOST value (contains commas). "
            "Pass SetEnv as separate KEY=VALUE items."
        )
    return output


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_image_override(
    task_def: dict[str, Any],
    image: str | None,
    container_name: str,
) -> None:
    if not image:
        return
    containers = task_def.get("containerDefinitions") or []
    for container in containers:
        if container.get("name") == container_name:
            container["image"] = image
            return
    if containers:
        containers[0]["image"] = image


def _apply_env_overrides(task_def: dict[str, Any], overrides: dict[str, str], container_name: str) -> None:
    if not overrides:
        return
    containers = task_def.get("containerDefinitions") or []
    target = None
    for container in containers:
        if container.get("name") == container_name:
            target = container
            break
    if target is None and containers:
        target = containers[0]
    if target is None:
        return

    existing = {item["name"]: item.get("value", "") for item in target.get("environment", []) if "name" in item}
    existing.update(overrides)
    target["environment"] = [{"name": k, "value": v} for k, v in existing.items()]


def _task_registration_payload(task_def: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = [
        "family",
        "taskRoleArn",
        "executionRoleArn",
        "networkMode",
        "containerDefinitions",
        "volumes",
        "placementConstraints",
        "requiresCompatibilities",
        "cpu",
        "memory",
        "tags",
        "pidMode",
        "ipcMode",
        "proxyConfiguration",
        "inferenceAccelerators",
        "ephemeralStorage",
        "runtimePlatform",
    ]
    payload = {k: task_def[k] for k in allowed_keys if k in task_def}
    if "family" not in payload:
        raise ValueError("Task definition JSON must include 'family'.")
    if "containerDefinitions" not in payload:
        raise ValueError("Task definition JSON must include 'containerDefinitions'.")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Register new ECS API task definition revision and force service deployment."
    )
    parser.add_argument(
        "--task-def-file",
        default="infra/ecs-api-task-definition.json",
        help="Path to ECS task definition template JSON.",
    )
    parser.add_argument("--cluster", default=os.getenv("ECS_CLUSTER", "rag-cluster"))
    parser.add_argument("--service", default=os.getenv("ECS_SERVICE", "rag-api-service-call"))
    parser.add_argument("--region", default=os.getenv("AWS_REGION", "ap-south-1"))
    parser.add_argument("--container-name", default="rag-api-service")
    parser.add_argument("--image", default="", help="Optional container image override.")
    parser.add_argument(
        "--set-env",
        action="append",
        default=[],
        help="Override or add env var in task definition container. Repeatable: --set-env KEY=VALUE",
    )
    parser.add_argument(
        "--wait-stable",
        action="store_true",
        help="Wait for ECS service to become stable after update.",
    )
    parser.add_argument(
        "--wait-timeout-seconds",
        type=int,
        default=900,
        help="Max wait time for service stabilization when --wait-stable is set.",
    )
    args = parser.parse_args()

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError, WaiterError
    except Exception as exc:
        print(f"ERROR: boto3 unavailable: {exc}", file=sys.stderr)
        return 2

    try:
        task_def = _load_json(args.task_def_file)
        _apply_image_override(task_def, args.image.strip() or None, args.container_name)
        env_overrides = _parse_set_env(args.set_env)
        _apply_env_overrides(task_def, env_overrides, args.container_name)
        payload = _task_registration_payload(task_def)
    except Exception as exc:
        print(f"ERROR: failed to prepare task definition payload: {exc}", file=sys.stderr)
        return 2

    ecs = boto3.client("ecs", region_name=args.region)

    try:
        register_resp = ecs.register_task_definition(**payload)
        task_arn = register_resp["taskDefinition"]["taskDefinitionArn"]
    except (ClientError, BotoCoreError) as exc:
        print(f"ERROR: register_task_definition failed: {exc}", file=sys.stderr)
        return 3

    try:
        update_resp = ecs.update_service(
            cluster=args.cluster,
            service=args.service,
            taskDefinition=task_arn,
            forceNewDeployment=True,
        )
        service_arn = update_resp["service"]["serviceArn"]
    except (ClientError, BotoCoreError) as exc:
        print(f"ERROR: update_service failed: {exc}", file=sys.stderr)
        return 4

    summary = {
        "status": "deployment_started",
        "region": args.region,
        "cluster": args.cluster,
        "service": args.service,
        "task_definition_arn": task_arn,
        "service_arn": service_arn,
        "image_override": args.image.strip() or None,
        "env_overrides": env_overrides,
    }
    print(json.dumps(summary, indent=2))

    if args.wait_stable:
        try:
            waiter = ecs.get_waiter("services_stable")
            attempts = max(1, args.wait_timeout_seconds // 15)
            waiter.wait(
                cluster=args.cluster,
                services=[args.service],
                WaiterConfig={"Delay": 15, "MaxAttempts": attempts},
            )
            print(json.dumps({"status": "stable", "cluster": args.cluster, "service": args.service}, indent=2))
        except WaiterError as exc:
            print(f"ERROR: service did not stabilize in time: {exc}", file=sys.stderr)
            return 5

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
