from  .state import State,Tag
from typing import Literal

def router(state:State)->Literal["diet","sport","sign","buy","sign_query","agent","__end__"]:
    tag:Tag = state["life_tag"]
    print("tag=======>",tag)
    if tag.is_question:
        for item in tag.subTags:
            match item.type:
                case "diet":
                    return "diet"
                case "sport":
                    return "sport"
                case "sign":
                    return "sign_query"
                case "buy":
                    return "buy"
                case _:
                    return "agent"
        return "agent"
    match tag.action:
        case "record":
            for item in tag.subTags:
                match item.type:
                    case "diet":
                        return "diet"
                    case "sport":
                        return "sport"
                    case "sign":
                        return "sign"
                    case "buy":
                        return "buy"
                    case _:
                        return "agent"
            return "agent"
        case _:
            return "agent"