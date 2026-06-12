# -*- coding: utf-8 -*-
"""Proto MainAction / AssistantAction 数值到简短英文标签，供 OpenCV putText 使用。"""

# senseAD MainAction（见 proto/dlp_raw_data.proto LinkAction.main_action_type）
MAIN_ACTION_LABEL = {
    255: "Unknown",  # 未知
    0: "Null",  # 无/空主动作
    1: "TurnLeft",  # 左转
    2: "TurnRight",  # 右转
    3: "SlightLeft",  # 左转微偏 / 略向左
    4: "SlightRight",  # 右转微偏 / 略向右
    5: "TurnHardLeft",  # 急左转
    6: "TurnHardRight",  # 急右转
    7: "UTurn",  # 掉头
    8: "Continue",  # 直行/保持
    9: "MergeLeft",  # 向左汇入
    10: "MergeRight",  # 向右汇入
    11: "EntryRing",  # 进入环岛
    12: "LeaveRing",  # 驶出环岛
    13: "Slow",  # 减速
    14: "PlugContinue",  # 接续/插入后继续（枚举名沿用数据侧）
    65: "EnterBuilding",  # 进入建筑物内道路
    66: "LeaveBuilding",  # 驶出建筑物
    67: "ByElevator",  # 经电梯（室内/立体交通）
    68: "ByStair",  # 经楼梯
    69: "ByEscalator",  # 经扶梯
    70: "Count",  # 枚举计数占位（非业务动作）
}

# AssistantAction（assist_action_type）
ASSIST_ACTION_LABEL = {
    0: "AssiNull",  # 无辅动作（标签名 Assi 为历史拼写）
    1: "EntryMain",  # 进入主路
    2: "EntrySideRoad",  # 进入辅路/侧路
    3: "EntryFreeway",  # 进入高速/快速路
    4: "EntrySlip",  # 进入匝道（slip road）
    5: "EntryTunnel",  # 进入隧道
    6: "EntryCenterBranch",  # 进入中间岔路
    7: "EntryRightBranch",  # 进入右侧岔路
    8: "EntryLeftBranch",  # 进入左侧岔路
    9: "EntryRightRoad",  # 进入右侧道路（专用道/右路）
    10: "EntryLeftRoad",  # 进入左侧道路（专用道/左路）
    11: "EntryMergeCenter",  # 汇入中间车道
    12: "EntryMergeRight",  # 向右汇入
    13: "EntryMergeLeft",  # 向左汇入
    14: "EntryMergeRightSild",  # 向右汇入侧路（Sild 为数据侧拼写，多指 side）
    15: "EntryMergeLeftSild",  # 向左汇入侧路
    16: "EntryMergeRightMain",  # 向右汇入主路
    17: "EntryMergeLeftMain",  # 向左汇入主路
    18: "EntryMergeRightRight",  # 向右再向右汇入（连续右转类）
    19: "EntryFerry",  # 进入渡口/轮渡衔接
    20: "LeftFerry",  # 离开渡口（向左驶离等语义，以数据为准）
    23: "AlongRoad",  # 沿当前道路
    24: "AlongSild",  # 沿辅路/侧路（Sild≈side）
    25: "AlongMain",  # 沿主路
    32: "ArriveExit",  # 到达出口
    33: "ArriveServiceArea",  # 到达服务区
    34: "ArriveTollGate",  # 到达收费站
    35: "ArriveWay",  # 到达道路节点/路段点
    36: "ArriveDestination",  # 到达目的地
    37: "ArriveChargingStation",  # 到达充电站
    48: "EntryRingLeft",  # 环岛左转进入
    49: "EntryRingRight",  # 环岛右转进入
    50: "EntryRingContinue",  # 环岛直行进入
    51: "EntryRingUturn",  # 环岛掉头进入
    52: "SmallRingNotCount",  # 小环岛不计入统计类（数据语义）
    64: "RightBranch1",  # 右侧分支 1
    65: "RightBranch2",  # 右侧分支 2
    66: "RightBranch3",  # 右侧分支 3
    67: "RightBranch4",  # 右侧分支 4
    68: "RightBranch5",  # 右侧分支 5
    69: "LeftBranch1",  # 左侧分支 1
    70: "LeftBranch2",  # 左侧分支 2
    71: "LeftBranch3",  # 左侧分支 3
    72: "LeftBranch4",  # 左侧分支 4
    73: "LeftBranch5",  # 左侧分支 5
    80: "EnterUline",  # 进入 U 形弯/掉头线
    90: "PassCrosswalk",  # 经过人行横道
    91: "PassOverpass",  # 经过天桥/上跨
    92: "PassUnderground",  # 经过地下通道
    93: "PassSquare",  # 经过广场
    94: "PassPark",  # 经过公园区域道路
    95: "PassStaircase",  # 经过阶梯段
    96: "PassLift",  # 经过升降机段
    97: "PassCableway",  # 经过索道
    98: "PassSkyChannel",  # 经过空中连廊
    99: "PassChannel",  # 经过渠道类构造
    100: "PassWalkroad",  # 经过步行道
    101: "PassBoatLine",  # 经过水运线路
    102: "PassSightSeeingLine",  # 经过观光线
    103: "PassSkidway",  # 经过滑道
    105: "PassLadder",  # 经过梯道
    106: "PassSlope",  # 经过坡道
    107: "PassBridge",  # 经过桥梁
    108: "PassFerry",  # 经过渡口路段
    109: "PassSubway",  # 经过地铁相关路段
    112: "SoonEnterBuilding",  # 即将进入建筑
    113: "SoonLeaveBuilding",  # 即将驶出建筑
    114: "EnterRoundabout",  # 进入环岛（辅动作语义）
    115: "LeaveRoundabout",  # 离开环岛
    116: "EnterPath",  # 进入小路/辅径
    117: "EnterInner",  # 进入内部路
    118: "EnterLeftBranchTwo",  # 进入左侧第二岔路
    119: "EnterLeftBranchThree",  # 进入左侧第三岔路
    120: "EnterRightBranchTwo",  # 进入右侧第二岔路
    121: "EnterRightBranchThree",  # 进入右侧第三岔路
    122: "EnterGasStation",  # 进入加油站
    123: "EnterHousingEstate",  # 进入小区内部路
    124: "EnterParkRoad",  # 进入园区/公园道路
    125: "EnterOverhead",  # 进入高架
    126: "EnterCenterBranchOverhead",  # 高架中间岔口进入
    127: "EnterRightBranchOverhead",  # 高架右侧岔口进入
    128: "EnterLeftBranchOverhead",  # 高架左侧岔口进入
    129: "AlongStraight",  # 沿直线段
    130: "DownOverhead",  # 下高架
    131: "EnterLeftOverhead",  # 从左侧上高架
    132: "EnterRightOverhead",  # 从右侧上高架
    133: "UpToBridge",  # 上至桥面
    134: "EnterParking",  # 进入停车场
    135: "EnterOverpass",  # 进入跨线桥/立交上层
    136: "EnterBridge",  # 进入桥梁段
    137: "EnterUnderpass",  # 进入下穿/桥洞
}


def format_main_action(v: int) -> str:
    return MAIN_ACTION_LABEL.get(int(v), f"M{int(v)}")


def format_assist_action(v: int) -> str:
    return ASSIST_ACTION_LABEL.get(int(v), f"A{int(v)}")


def truncate_label(s: str, max_len: int = 52) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."
