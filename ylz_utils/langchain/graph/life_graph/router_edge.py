from  .state import State,Tag

def router(state:State):
    tag:Tag = state["life_tag"]
    print("tag=",tag)
    if tag.is_question:
        return "agent"
    match tag.action:
        case "record":
            match tag.type:
                case "diet":
                    return "diet"
                case "sport":
                    return "sport"
                case _:
                    return "agent"
        case _:
            return "agent"