#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

for lang in ara eng fin tur heb rus
    do
        sleep 2
        for scenario in generation judge
            do
                for system_method in direct cot
                    do
                        sleep 2
                        for model in aya-expanse-32b euroLLM gemma-3-27b-it phi-4-mini-instruct qwen-2.5-32B-instruct qwen-3-32B-no-think gemini-2.0-flash-lite gemini-2.0-flash
                            do
                                python scripts/run_inference.py model=$model system_method=$system_method --lang_code $lang --scenario $scenario
                            done
                    done
            done
    done
