from langchain_community.document_loaders import UnstructuredFileLoader
from tqdm import tqdm
from ylz_utils import LangchainLib
from langchain_community.vectorstores import FAISS

def start_rag(langchainLib:LangchainLib,args):
    embedding_key = args.embedding_key
    embedding_model = args.embedding_model
    rag_indexname = args.rag_indexname or "faiss:index"
    url = args.url
    pptx = args.pptx
    docx = args.docx
    pdf = args.pdf
    glob = args.glob
    message = args.message
    fake_size = args.fake_size
    
    if (not url and not pptx and not docx and not pdf and not glob) and  (not message):
        print(f"1、指定url/pptx/docx:系统将文档下载切片后向量化到{rag_indexname}数据表\n2、指定message:系统将从{rag_indexname}数据表中搜索相关的两条记录。\n您需要至少指定url和message中的一个参数.")
        return

    if embedding_key or embedding_model or fake_size:
        embedding = langchainLib.get_embedding(embedding_key,embedding_model,fake_size=fake_size)
    else:
        embedding = None

    if (url or pptx or docx or pdf or glob):
        docs = start_loader(langchainLib,args)
        print("#"*60)
        if not docs:
            return
        batch = args.batch

        vectorstore = langchainLib.vectorstoreLib.get_store_with_provider_and_indexname(rag_indexname,embedding=embedding)
        
        if url and rag_indexname:
            ##### create vectorestore
            # url = "https://python.langchain.com/v0.2/docs/concepts/#tools"
            # faiss_dbname = "langchain_docs.faiss"
            print("result:",[{"doc_len":len(doc['doc'].page_content),"doc_blocks":len(doc['blocks'])} for doc in docs])
            for doc in tqdm(docs,desc="页面",colour="#6666ff"):
                blocks = doc['blocks']
                ids = langchainLib.vectorstoreLib.add_docs(vectorstore,blocks,batch=batch)                    
                print("ids:",ids)
        else:
            ids = langchainLib.vectorstoreLib.add_docs(vectorstore,docs,batch=batch)                    
            print("ids:",ids)   
    
        if isinstance(vectorstore,FAISS):
            langchainLib.vectorstoreLib.faissLib.save(vectorstore)

        if message and rag_indexname:           
            print("v--->",langchainLib.vectorstoreLib.search_with_score(message,vectorstore,k=4))
    
    ###### have bug when poetry add sentence_transformers   
    #v1 = langchainLib.get_huggingface_embedding()
    #print("huggingface BGE:",v1)

    ###### tet google embdeeding
    # embed = langchainLib.get_embedding("EMBEDDING.GEMINI")
    # docs = [Document("I am a student"),Document("who to go to china"),Document("this is a table")]
    # vectorestore = langchainLib.vectorstoreLib.faissLib.create_from_docs(docs,embedding=embed)
    # langchainLib.vectorstoreLib.faissLib.save("test.faiss",vectorestore,index_name="gemini")

def start_loader(langchainLib:LangchainLib,args):
    url = args.url
    depth = args.depth
    docx_file = args.docx
    pptx_file = args.pptx
    pdf_file = args.pdf
    glob = args.glob
    chunk_size = args.size or 512
    only_one = list(filter(lambda x: x,[url,docx_file,pptx_file,pdf_file,glob]))
    if len(only_one) != 1:
        print(f"请指定url,docx,pptx,pdf,glob其中的一个")
        return 
    if url:
        result = langchainLib.load_url_and_split_markdown(url = url,max_depth = depth, chunk_size=chunk_size)
        print("result:",[{"doc_len":len(doc['doc'].page_content),"doc_blocks":len(doc['blocks']),"metadata":doc['metadata']} for doc in result])
    elif docx_file:
        result = langchainLib.documentLib.docx.load_and_split(docx_file,chunk_size=chunk_size)
        print(result)
    elif pptx_file:
        result = langchainLib.documentLib.pptx.load_and_split(pptx_file,chunk_size=chunk_size)
        print(result)
    elif pdf_file:
        result = langchainLib.documentLib.pdf.load_and_split(pdf_file,chunk_size=chunk_size)
        print(result)
    elif glob:
        loader_cls = UnstructuredFileLoader
        if glob.find(".docx")>=0:
            loader_cls = langchainLib.documentLib.docx.loader
        elif glob.find(".pptx")>=0:
            loader_cls = langchainLib.documentLib.pptx.loader
        elif glob.find(".pdf")>=0:
            loader_cls = langchainLib.documentLib.pdf.loader
        loader  = langchainLib.documentLib.dir.loader(".",glob=glob,show_progress=True,loader_cls=loader_cls)
        result  = loader.load_and_split(langchainLib.splitterLib.get_textsplitter(chunk_size=chunk_size,chunk_overlap=0))
        print(result)
    return result
