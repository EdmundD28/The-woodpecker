import json
import sys
from datetime import datetime
from pathlib import Path

from protocol import validate_record


OUTPUT_DIRECTORY = Path("collected_data")


def create_output_path() -> Path:
    """创建不会重复的数据文件名。"""

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    return OUTPUT_DIRECTORY / (
        f"session_{timestamp}.jsonl"
    )


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = create_output_path()

    valid_count = 0
    invalid_count = 0
    sample_ids = set()

    print(
        "采集工具已启动，正在等待数据……",
        file=sys.stderr,
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
        newline="\n",
    ) as output_file:

        for line_number, input_line in enumerate(
            sys.stdin,
            start=1,
        ):
            input_line = input_line.strip()

            if not input_line:
                continue

            try:
                record = json.loads(input_line)

                validate_record(record)

                sample_id = record["sample_id"]

                if sample_id in sample_ids:
                    raise ValueError(
                        f"样本编号重复: {sample_id}"
                    )

                sample_ids.add(sample_id)

                saved_line = json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

                output_file.write(saved_line + "\n")
                output_file.flush()

                valid_count += 1

                print(
                    "已保存: "
                    f"{sample_id}，"
                    f"位置=P{record['impact_point_id']:02d}，"
                    f"采样点数={record['sample_count']}",
                    file=sys.stderr,
                )

            except json.JSONDecodeError as error:
                invalid_count += 1

                print(
                    f"第{line_number}行不是有效JSON: "
                    f"{error}",
                    file=sys.stderr,
                )

            except ValueError as error:
                invalid_count += 1

                print(
                    f"第{line_number}行协议验证失败: "
                    f"{error}",
                    file=sys.stderr,
                )

    if valid_count == 0:
        output_path.unlink(missing_ok=True)

        print(
            "没有收到有效记录，未保存数据文件。",
            file=sys.stderr,
        )

        return

    print(
        "",
        file=sys.stderr,
    )

    print(
        "采集结束",
        file=sys.stderr,
    )

    print(
        f"有效记录: {valid_count}",
        file=sys.stderr,
    )

    print(
        f"无效记录: {invalid_count}",
        file=sys.stderr,
    )

    print(
        f"保存位置: {output_path.resolve()}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()