# Copyright 2015 gRPC authors.
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
"""The Python implementation of the gRPC route guide client."""

from __future__ import print_function

import grpc
import logging
from . import pascal_voc_pb2, pascal_voc_pb2_grpc

from pathlib import Path
from typing import List
from kazoo.client import KazooClient


class ZooKeeper:
    def __init__(self) -> None:
        self.kz = KazooClient(
            hosts="local.inhandplus.com:2181,local.inhandplus.com:2182,local.inhandplus.com:2183"
        )
        self.kz.start()

    def get(self, node):
        return self.kz.get(node)

    def get_children(self, node):
        return self.kz.get_children(node)


class ZKLB:
    def __init__(self, zookeeper: ZooKeeper, node_info="/grpc/server") -> None:
        self.zk = zookeeper
        self.node_info = "/grpc/server"

    def select_server(self):
        return [
            str(Path(self.node_info) / node)
            for node in self.zk.get_children(self.node_info)
        ]

    def compare_queue(self, nodes: List[str]):
        server_info = {
            Path(node).name: self.zk.get(node)[0].decode("utf-8") for node in nodes
        }
        return dict(sorted(server_info.items(), key=lambda item: item[1])).keys()


def chunk_generator(filepath, obj, stride):
    with open(filepath, "rb") as f:
        info = pascal_voc_pb2.DetectFile(filename=filepath, object=obj, stride=stride)
        yield pascal_voc_pb2.FileReq(info=info)
        while True:
            piece = f.read(1024 * 1024)
            if len(piece) == 0:
                return
            yield pascal_voc_pb2.FileReq(buffer=piece)


def prediction_file(stub, filepath, obj, stride=1):
    chunks = chunk_generator(filepath, obj, stride)
    feature = stub.PredictionFile(chunks)
    print(feature)
    return feature


def prediction_blob(stub, project_name, filename, obj_name, stride=1):
    req = pascal_voc_pb2.BlobReq(
        project=project_name,
    )
    detect_file = req.detect_file
    detect_file.object = obj_name
    detect_file.filename = filename
    detect_file.stride = stride

    video_result = stub.PredictionBlob(req)

    for img in video_result.image:
        for obj in img.bbox:
            print(img.index, obj.cls, obj.xmin, obj.ymin, obj.xmax, obj.ymax, obj.score)
    return len(video_result.image)


def check_blob(stub, project_name, filename):
    result = stub.CheckBlob(
        pascal_voc_pb2.BlobFile(project=project_name, filename=filename)
    )
    return result


def run_test():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    host = "local.inhandplus.com"
    port = 50050
    with open(Path(__file__).parent / Path("star_inhandplus_com.crt"), "rb") as f:
        trusted_certs = grpc.ssl_channel_credentials(f.read())

    with grpc.secure_channel(f"{host}:{port},", trusted_certs) as channel:
        stub = pascal_voc_pb2_grpc.PascalVOCLabelStub(channel)
        print("-------------- PredictionFile --------------")
        prediction_file(stub, "tests/testvideo.mp4", obj="eyedrop")

        print("-------------- PredictionBlob --------------")
        prediction_blob(
            stub,
            "merck",
            "2021-03-21/20IHPA00110A_210321_104910_D487D4B2FA67.mp4",
            "eyedrop",
        )
        print("-------------- Checkfile --------------")
        checked = check_blob(
            stub, "merck", "2021-03-21/20IHPA00110A_210321_104910_D487D4B2FA67.mp4"
        )
        print("-------------- Total --------------")
        filenames = [
            "2021-03-21/20IHPA00110A_210321_104910_D487D4B2FA67.mp4",
            "tests/testvideo.mp4",
        ]
        objname = "eyedrop"
        project = "merck"
        for filename in filenames:
            checked = check_blob(stub, project, filename)
            if checked:
                prediction_blob(
                    stub,
                    project,
                    filename,
                    objname,
                )
            else:
                prediction_file(stub, filename, obj=objname)
        channel.close()


def run(filename, project, objname, stride=5):
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    lb = ZKLB(ZooKeeper())
    servers = lb.compare_queue(lb.select_server())
    with open(Path(__file__).parent / "ssl/star_inhandplus_com.crt", "rb") as f:
        trusted_certs = grpc.ssl_channel_credentials(f.read())

    for server in servers:
        with grpc.secure_channel(server, trusted_certs) as channel:
            stub = pascal_voc_pb2_grpc.PascalVOCLabelStub(channel)
            try:
                checked = check_blob(stub, project, filename)
                if checked.res:
                    prediction_blob(
                        stub,
                        project,
                        filename,
                        objname,
                        stride=stride,
                    )
                else:
                    results = prediction_file(
                        stub, filename, obj=objname, stride=stride
                    )
                    break
            except:
                results = prediction_file(stub, filename, obj=objname, stride=stride)
                break
            channel.close()
    return results


if __name__ == "__main__":
    logging.basicConfig()
    # run_test()
