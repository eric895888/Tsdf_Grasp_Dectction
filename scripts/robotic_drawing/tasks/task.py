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

class Task:
    def __init__(self, name=None, description=None, func=None):
        self.name = name
        self.description = description
        self.func = func

    def run(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    
class TaskManager:
    def __init__(self):
        self.tasks = {}
    
    def add_task(self, task):
        self.tasks[task.name] = task

    def run_task(self, task_name, *args, **kwargs):
        return self.tasks[task_name].run(*args, **kwargs)
    
    def get_task(self, task_name):
        return self.tasks[task_name]
    
    def get_tasks(self):
        return self.tasks

class SubTask:
    def __init__(self, name=None, description=None, func=None):
        self.name = name
        self.description = description
        self.func = func

    def run(self, *args, **kwargs):
        return self.func(*args, **kwargs)