# Copyright 2024 tc-haung
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def micro_meter_to_mili_meter(micro_meter):
    assert micro_meter >= 0, "micro_meter should be non-negative"
    return micro_meter / 1000.0


def mili_meter_to_meter(mili_meter):
    assert mili_meter >= 0, "mili_meter should be non-negative"
    return mili_meter / 1000.0


def micro_meter_to_meter(micro_meter):
    assert micro_meter >= 0, "micro_meter should be non-negative"
    return micro_meter / 1000000.0
