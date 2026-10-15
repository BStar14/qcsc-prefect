import argparse
import asyncio

from sbd.qfw_resource import QFwResource


async def main():
    parser = argparse.ArgumentParser(description="Create the QFwResource Prefect block.")
    parser.add_argument(
        "--device-id",
        required=True,
        help="Device ID configured in device-access.yaml.",
    )
    parser.add_argument(
        "--device-access-config",
        default="/workspace/qfw-container-base/device-access.yaml",
        help="Path to the QFw device-access configuration.",
    )
    parser.add_argument(
        "--block-name",
        default="qfw-runner",
        help="Prefect block name.",
    )

    args = parser.parse_args()

    block = QFwResource(
        device_id=args.device_id,
        device_access_config=args.device_access_config,
    )

    await block.save(
        args.block_name,
        overwrite=True,
    )

    print(f"Created QFwResource block: {args.block_name}")
    print(f"Device ID: {args.device_id}")


if __name__ == "__main__":
    asyncio.run(main())
