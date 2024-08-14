def _get_llm(config, node_key):
    graphLib = config['configurable']['graphLib'].engineer_graph
    llm = graphLib.get_node_llm(node_key)    
    return llm