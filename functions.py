
def get_embedding(text, endpoint_name="databricks-bge-large-en"):
    """
    埋め込みモデルでテキストをベクトル化する関数
    """
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    response = deploy_client.predict(
        endpoint=endpoint_name,
        inputs={"input": [text]}
    )
    return response.data[0]['embedding']

def search_knowledge_base(query_text, num_results=3):
    """
    質問文を受け取り、Vector Searchで関連ドキュメントを検索して、
    その内容を結合した文字列（コンテキスト）を返す。
    --- Vector Searchでナレッジを検索する関数 ---
    """
    # インデックスを取得
    vsc = VectorSearchClient()
    index = vsc.get_index(
        endpoint_name="my_vector_search_endpoint", # ご自身の環境のエンドポイント名
        index_name="genai.vectordb.my_document_index" # ご自身の環境のインデックス名
    )
    
    # 質問文をベクトル化
    query_vector = get_embedding(query_text)
    
    # 類似検索を実行
    results = index.similarity_search(
        query_vector=query_vector,
        num_results=num_results,
        columns=["text_content"] # 回答生成に必要なテキスト内容の列を指定
    )
    
    # 検索結果のテキスト部分を抽出
    docs = results.get('result', {}).get('data_array', [])
    if not docs:
        return "" # 結果がない場合は空文字を返す
    
    # 複数のドキュメントを一つの文字列に結合
    context = "\n\n".join([doc[0] for doc in docs])
    return context

def answer_question_with_rag(question, chat_model_endpoint="databricks-gpt-oss-20b"):
    """
    質問を受け取り、RAGパイプライン全体を実行して最終的な回答を生成する。
    """
    # --- RAGを実行して回答を生成するメイン関数 ---
    #以下のようにprompt_templateを指定すると、RAGの情報だけで答える。
    #あなたは社内情報を答える優秀なアシスタントです。
    #以下の「コンテキスト」情報だけを厳密に利用して、ユーザーの「質問」に日本語で回答してください。
    #コンテキストに記載されていない事柄については、絶対に回答に含めず、「その情報は見つかりませんでした」と答えてください。
    #
    #以下のように設定すると、外部のLLMも使って答える
    #あなたは社内情報も答えられる優秀なアシスタントです。
    #以下の「コンテキスト」情報を利用して、ユーザーの「質問」に日本語で回答してください。
    # 1. Vector Searchで関連情報を検索
    #print(f"Question: {question}")
    context = search_knowledge_base(question)
    #print(f"VectorDB query result:\n {context}")
    if not context:
        return "申し訳ありませんが、関連する情報が見つかりませんでした。"

    # 2. LLMに渡すためのプロンプトをテンプレートを使って作成
    prompt_template = f"""
あなたは社内情報を答える優秀なアシスタントです。
以下の「コンテキスト」情報だけを厳密に利用して、ユーザーの「質問」に日本語で回答してください。
コンテキストに記載されていない事柄については、絶対に回答に含めず、「その情報は見つかりませんでした」と答えてください。

【コンテキスト】
{context}

【質問】
{question}

【回答】
"""


    # 3. Foundation Model API (LLM) を呼び出して回答を生成
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    response = deploy_client.predict(
        endpoint=chat_model_endpoint,
        inputs={
            "messages": [
                {
                    "role": "user",
                    "content": prompt_template
                }
            ],
            "max_tokens": 500 # 回答の最大長
        }
    )
    
    # 4. LLMからの回答を返す
    return response.choices[0]['message']['content']

def answer_question_with_graph(question,graph_context, chat_model_endpoint="databricks-gpt-oss-20b"):
    """
    質問を受け取り、Vector DBとGraph DBを検索した結果を使って回答を生成する。
    """
    #print(f"Question: {question}\n{graph_context}")

    # 1. Vector Searchで関連情報を検索
    vector_context = search_knowledge_base(question)
    #print(f"VectorDB query result:\n {vector_context}")
    if not vector_context:
        return "申し訳ありませんが、関連する情報が見つかりませんでした。"

    # 2. LLMに渡すためのプロンプトをテンプレートを使って作成
    prompt_template = f"""
あなたは社内情報を答える優秀なアシスタントです。
以下のコンテキストに基づいてユーザーの「質問」に日本語で回答してください。
コンテキストに記載されていない事柄については、絶対に回答に含めず、「その情報は見つかりませんでした」と答えてください。

【コンテキスト】
情報: {vector_context}
グラフデータ: {graph_context}

【質問】
{question}

【回答】
"""
    # 3. Foundation Model API (LLM) を呼び出して回答を生成
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    response = deploy_client.predict(
        endpoint=chat_model_endpoint,
        inputs={
            "messages": [
                {
                    "role": "user",
                    "content": prompt_template
                }
            ],
            "max_tokens": 1000 # 回答の最大長
        }
    )
    
    # 4. LLMからの回答を返す
    return response.choices[0]['message']['content']

def get_node_from_question(question,type, chat_model_endpoint="databricks-gpt-oss-20b"):
    """
    質問を受け取り、検索したいNodeを生成する
    """
    # LLMに渡すためのプロンプトをテンプレートを使って作成
    prompt_template = f"""
あなたはGraph DB用のEdgeとNodeを抽出するアシスタントです。
以下の質問から、Nodeとなる単語を抽出してください。
その際コンテキストに指定されているtypeのNodeのみ抽出してください。
質問が日本語の場合、単語に分解してから抽出してください。
回答は、Nodeのみを出力してください。

【コンテキスト】
{type}

【質問】
{question}

【回答】
"""
    # 3. Foundation Model API (LLM) を呼び出して回答を生成
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    response = deploy_client.predict(
        endpoint=chat_model_endpoint,
        inputs={
            "messages": [
                {
                    "role": "user",
                    "content": prompt_template
                }
            ],
            "max_tokens": 500 # 回答の最大長
        }
    )
    
    # 4. LLMからの回答を返す
    return response.choices[0]['message']['content']

# GraphMLを使ってGraph DBをシミュレートする

def load_graphml(graphml_file_path):
    """
    GraphMLファイルを読み込み、様々な条件でノードを検索する関数。

    Args:
        graphml_file_path (str): GraphMLファイルのパス。
    """
    if not os.path.exists(graphml_file_path):
        ic(f"エラー: ファイル '{graphml_file_path}' が見つかりません。")
        return

    try:
        # GraphMLファイルを読み込む
        G = nx.read_graphml(graphml_file_path)
        #print(f"グラフ '{graphml_file_path}' を正常に読み込みました。")
        #print(f"ノード数: {G.number_of_nodes()}, エッジ数: {G.number_of_edges()}")
        return G

    except nx.NetworkXError as e:
        ic(f"GraphMLファイルの読み込み中にエラーが発生しました: {e}")
        return
    except Exception as e:
        ic(f"予期せぬエラーが発生しました: {e}")
        return
    
def find_node(G,node_id_to_find):
    """
    GraphMLオブジェクトGから、node idを検索する関数。
    """
    if node_id_to_find in G.nodes:
        node_data = G.nodes[node_id_to_find]
        return node_data
    else:
        ic(f"ノード '{node_id_to_find}' が見つかりませんでした。")
        return
    
def find_node_by_label(G,target_label):
    """
    GraphMLオブジェクトGから、labelを検索する関数。
    """
    found_nodes = []
    for node_id, data in G.nodes(data=True):
        if data.get("label") == target_label:
            found_nodes.append((node_id, data))
    return found_nodes

def find_node_by_type_and_value(G,target_type,target_value):
    """
    GraphMLオブジェクトGから、typeとvalueで検索する関数。
    """
    found_nodes = []
    for node_id, data in G.nodes(data=True):
        if data.get("type") == target_type and data.get("value") == target_value:
            found_nodes.append((node_id, data))
    return found_nodes

def get_edges_for_node(G, target_node_id):
    """
    GraphMLオブジェクトGから、node idを指定して、つながっているEdgeとNodeを検索する
    """
    if target_node_id not in G.nodes:
        print(f"\nエラー: ノードID '{target_node_id}' はグラフに存在しません。")
        return [], [], []

    #    print(f"\n--- ノード '{target_node_id}' のエッジを抽出中 ---")

    # 1. そのノードから出るエッジ (Out-edges)
    # G.out_edges(node_id, data=True) は有向グラフでのみ使用可能
    # G.edges(node_id, data=True) は、DiGraphではout_edgesと同じ、Graphではincident_edgesと同じ

    out_edges = list(G.out_edges(target_node_id, data=True))

    # 2. そのノードに入るエッジ (In-edges)
    # G.in_edges(node_id, data=True) は有向グラフでのみ使用可能

    in_edges = list(G.in_edges(target_node_id, data=True))

    # 3. そのノードに接続する全てのエッジ (Incident edges)
    # 有向グラフの場合、out_edges と in_edges の結合。
    # 無向グラフの場合、G.edges(node_id, data=True) で直接取得できる。
    # ここでは有向グラフなので、両方を結合。
 
    all_incident_edges = out_edges + in_edges
 
    #print(f"\nノード '{target_node_id}' に接続する全てのエッジ ({len(all_incident_edges)}個):")
    # if all_incident_edges:
    #     for u, v, data in all_incident_edges:
    #         print(f"  - {u} {'->' if G.is_directed() else '--'} {v}, 属性: {data}")
    # else:
    #     print("  なし")

    return all_incident_edges        
 #   return out_edges, in_edges, all_incident_edges     

 #LLMの出力からproductに関連するNodeを検出する
    
def get_node_from_llmoutput(llmoutput):

    # Find the dict with type 'text' and parse its 'text' field as JSON

    for item in llmoutput:
        if item.get('type') == 'text':
            try:
                nodes_json = item.get("text")
                nodes_dict = json.loads(nodes_json)
                return nodes_dict
            except json.JSONDecodeError as e:
                return

def get_graph_context(g,findword):
    """
    graphml g から findwordをlabel(name)で検索し、そのノードに接続しているノードを検索する
    """
    # 1

    # Graphmlファイルを読み込んで、オブジェクトを生成
    #graphml_file = "/Workspace/Users/keitan1@fpt.com/VectorDB/output_graph.graphml"
    #g=load_graphml(graphml_file)
    # 検索したいワードのidを取り出し
    result=find_node_by_label(g,findword)
    id=result[0][0]

    #idで、Edgeを検索する
    result=get_edges_for_node(g,id)

    results=[]
    for index, item in enumerate(result): # enumerateでインデックスも取得
        #print(f"{index}: {item}")
        fromid=item[0]
        toid=item[1]
        from_label=g.nodes[fromid]['label']
        to_label=g.nodes[toid]['label']
        relation=item[2]['relationship']
        results.append(f"'{from_label.strip()}' {relation.replace('_',' ')} '{to_label.strip()}'")

    return results