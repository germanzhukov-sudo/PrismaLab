#!/usr/bin/env python3
"""Скрипт для получения себестоимости паков из Astria API."""
import asyncio
import json
import os
import sys

# Добавляем корень проекта в path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from prismalab.astria_client import _get_pack, _timeout_s


def main():
    api_key = (os.getenv("PRISMALAB_ASTRIA_API_KEY") or os.getenv("ASTRIA_API_KEY") or "").strip()
    if not api_key:
        print("PRISMALAB_ASTRIA_API_KEY не задан", file=sys.stderr)
        return 1

    pack_ids = [264, 2156, 884, 268]
    timeout = _timeout_s(30.0)

    for pack_id in pack_ids:
        try:
            data = _get_pack(api_key=api_key, pack_id=pack_id, timeout_s=timeout)
            title = data.get("title") or data.get("slug") or f"Pack {pack_id}"
            costs = data.get("costs")
            cost_by_class = data.get("cost_by_class")
            default_cost = data.get("default_cost")
            num_images_by_class = data.get("num_images_by_class")
            default_num_images = data.get("default_num_images")

            print(f"\n=== Pack {pack_id}: {title} ===")
            print(f"  costs (raw): {json.dumps(costs, indent=4) if costs else 'None'}")
            print(f"  cost_by_class: {json.dumps(cost_by_class, indent=4) if cost_by_class else 'None'}")
            print(f"  default_cost: {json.dumps(default_cost, indent=4) if default_cost else 'None'}")
            print(f"  num_images_by_class: {num_images_by_class}")
            print(f"  default_num_images: {default_num_images}")
        except Exception as e:
            print(f"\n=== Pack {pack_id}: ОШИБКА ===\n  {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
