from  .state import State,Tag

def router(state:State):
    tag:Tag = state["life_tag"]
    print("tag=======>",tag)
    if tag.is_question:
        return "agent"
    match tag.action:
        case "record":
            for item in tag.subTags:
                match item.type:
                    case "diet":
                        return "diet"
                    case "sport":
                        return "sport"
                    case _:
                        return "agent"
            return "agent"
        case _:
            return "agent"