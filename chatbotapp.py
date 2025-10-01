import streamlit as st
import requests
import base64
import time

# DatabricksのワークスペースURLとトークン
workspace_url = "https://adb-3805281180121377.17.azuredatabricks.net"
token = "dapi06d3afc274ef9e04877b7ace110c024d"

headers = {
    "Authorization": f"Bearer {token}"
}

# ジョブID
job_ids = {
    "ベクトルのみ": 991771428805335,
    "ベクトルの検索結果とグラフの検索結果を合わせてLLMに問い合わせる(Vector first)": 517140104159628,
    "グラフの結果を使って、ベクトルに問い合わせた結果を、LLMに問い合わせる(Graph first)": 617814454526824
}

prompt_types = {
    "外部LLMの情報を使わない": "no-llm",
    "外部LLMの情報を使う": "with-llm"
}


st.title("Databricks RAG+Graph Chatbot Demo")

question = st.text_area("質問を入力してください")
prompt_option = st.selectbox("LLMのプロンプトタイプを選択", list(prompt_types.keys()))
type_option = st.selectbox("検索タイプを選択", list(job_ids.keys()))

if st.button("実行"):
    if not question.strip():
        st.warning("質問を入力してください。")
    else:
        with st.spinner("API実行中..."):
            # 質問をbase64エンコード
            encoded_question = base64.b64encode(question.encode("utf-8")).decode("ascii")
            job_id = job_ids[type_option]
            endpoint = f"{workspace_url}/api/2.2/jobs/run-now"
            payload = {
                "job_id": job_id,
                "notebook_params": {"question": encoded_question,"prompt_type":prompt_types[prompt_option]}
            }
            response = requests.post(endpoint, headers=headers, json=payload)
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as err:
                st.error(f"ジョブの実行に失敗しました: {err}")
                st.stop()

            parent_run_id = response.json()['run_id']
            st.info(f"ジョブ実行開始。run_id: {parent_run_id}")

            # ジョブの完了を待つ
            status_endpoint = f"{workspace_url}/api/2.2/jobs/runs/get?run_id={parent_run_id}"
            start_time = time.time()
            while True:
                status_response = requests.get(status_endpoint, headers=headers)
                status_info = status_response.json()
                state = status_info.get("state", {})
                life_cycle_state = state.get("life_cycle_state")
                st.write(f"現在の状態: {life_cycle_state}")

                if life_cycle_state in ["TERMINATED", "SKIPPED", "INTERNAL_ERROR"]:
                    break
                if time.time() - start_time > 360:
                    st.error("タイムアウトしました。（タイムアウト5分）")
                    st.stop()
                time.sleep(10)

            # 結果取得
            response = requests.get(f"{workspace_url}/api/2.2/jobs/runs/get?run_id={parent_run_id}", headers=headers)
            run_info = response.json()
            tasks = run_info.get('tasks', [])
            for task in tasks:
                task_run_id = task['run_id']
                output_response = requests.get(f"{workspace_url}/api/2.2/jobs/runs/get-output?run_id={task_run_id}", headers=headers)
                task_output = output_response.json().get('notebook_output', {}).get('result')
                st.subheader(f"{task['task_key']}の出力")
                #st.code(task_output, language="text", line_numbers=False)  # 折り返し表示

                # 折り返しを強制したい場合はst.markdownを使う
                st.markdown(
                    f"<div style='white-space:pre-wrap; word-break:break-all; font-family:monospace;'>{task_output}</div>",
                    unsafe_allow_html=True
                )