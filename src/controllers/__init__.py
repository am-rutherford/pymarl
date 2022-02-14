REGISTRY = {}

from .basic_controller import BasicMAC
REGISTRY["basic_mac"] = BasicMAC

from .zoo_controller import ZooMAC
REGISTRY["zoo_mac"] = ZooMAC