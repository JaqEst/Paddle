#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

from test_collective_api_base import TestDistBase

import paddle

paddle.enable_static()


class TestProcessGroupBackendRegistry(unittest.TestCase):
    def test_registered_backend_factory_is_used(self):
        backend = "dummy_backend"
        store = object()
        options = {"endpoint": "127.0.0.1:1234"}
        sentinel = object()
        calls = {}

        def factory(
            store,
            rank,
            world_size,
            group_id,
            pg_options,
            **kwargs,
        ):
            calls["store"] = store
            calls["rank"] = rank
            calls["world_size"] = world_size
            calls["group_id"] = group_id
            calls["pg_options"] = pg_options
            calls["kwargs"] = kwargs
            return sentinel

        try:
            paddle.distributed.register_process_group_backend(backend, factory)
            pg = paddle.distributed.collective._new_process_group_impl(
                backend,
                store,
                rank=1,
                world_size=4,
                group_name="test_group",
                pg_options=options,
                group_id=7,
            )

            self.assertIs(pg, sentinel)
            self.assertIs(calls["store"], store)
            self.assertEqual(calls["rank"], 1)
            self.assertEqual(calls["world_size"], 4)
            self.assertEqual(calls["group_id"], 7)
            self.assertIs(calls["pg_options"], options)
            self.assertEqual(calls["kwargs"]["group_name"], "test_group")
        finally:
            paddle.distributed.unregister_process_group_backend(backend)


class TestCollectiveAllreduceAPI(TestDistBase):
    def _setup_config(self):
        pass

    def test_allreduce_nccl(self):
        self.check_with_place(
            "collective_allreduce_new_group_api.py", "allreduce", "nccl"
        )


if __name__ == '__main__':
    unittest.main()
