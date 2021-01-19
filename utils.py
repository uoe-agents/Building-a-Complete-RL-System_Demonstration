import numpy as np


def act_to_str(act: int) -> str:
    """
    Map actions (of Taxi-v3 environment) to interpretable symbols corresponding to directions

    :param act (int): action to map to string
    :return (str): interpretable action name
    """
    if act == 0:
        return "S"
    elif act == 1:
        return "N"
    elif act == 2:
        return "E"
    elif act == 3:
        return "W"
    elif act == 4:
        return "P"
    elif act == 5:
        return "D"
    else:
        raise ValueError("Invalid action value")

def visualise_q_table(q_table):
    """
    Print q_table in human-readable format

    :param q_table (Dict): q_table in form of a dict mapping (observation, action) pairs to
        q-values
    """
    for key in sorted(q_table.keys()):
        obs, act = key
        act_name = act_to_str(act)
        q_value = q_table[key]
        print(f"Pos={obs}\tAct={act_name}\t->\t{q_value}")
