#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

for lang in ara eng fin tur heb rus
    do
        sleep 2
        for scenario in generation judge
            do
                for system_method in think
                    do
                        sleep 2
                        for model in  qwen-3-32B-think
                            do
                                python scripts/run_inference.py model=$model system_method=$system_method --lang_code $lang --scenario $scenario
                            done
                    done
            done
    done
