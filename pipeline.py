from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain  

def run_research_pipeline(topic : str) -> dict:
    state = {}
    
    # Step 1: Search for information on the topic
    
    print("\n"+"="*50)
    print("Step 1 - Search agent is working ....")
    print("="*50)
    
    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages" : [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    
    state['search_results'] = search_result['messages'][-1].content

    print("\n Search Result\n", state['search_results']) 
    
    # Step 2: Read and scrape content from the URLs found   
    
    print("\n"+"="*50)
    print("Step 2 - Reader agent is scraping top resources ....")
    print("="*50)
    
    reader_agent = build_search_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
        f"Based.on.the following search results about '{topic}',"
        f"pick the most relevant URL and scrape it for deeper content. \n\n"
        f"Search Results:\n{state['search_results'][:800] }"
    )]
})
    
    state['scraped_content'] = reader_result['messages'][-1].content

    print("\n Scraped Content\n", state['scraped_content']) 
    
    # Step 3: Write a research report based on the gathered information
    
    print("\n"+"="*50)
    print("Step 3 - Writer chain is generating a research report ....")
    print("="*50)
    
    research_combined = (
        f"SEARCH RESULTS : \n {'search_results'} \n\n"
        f"SCRAPED CONTENT : \n {'scraped_content'}"
    )
    
    state['report'] = writer_chain.invoke({
        "topic" : topic,
        "research" : research_combined
    })
    
    print("\n Research Report\n", state['report'])
    
    # Step 4: Critique the research report
    
    print("\n"+"="*50)
    print("Step 4 - Critic chain is evaluating the research report ....")
    print("="*50)
    
    state["feedback"] = critic_chain.invoke({
        "report" : state['report']
    })
    
    print("\n Critic report\n",state['feedback'])
    
    return state

if __name__ == "__main__":
    topic = input("\n Enter a Research topic : ")
    run_research_pipeline(topic)
