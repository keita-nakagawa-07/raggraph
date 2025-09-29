import base64
import sys
import time
import urllib
import requests

if len(sys.argv) != 3:
    print(len(sys.argv))
    print("Usage: python apitest.py <question> <type (1:vector-only,2:vector-first,3:graph-first)>")
    sys.exit(1)

question = sys.argv[1]
type = sys.argv[2]

print(f"質問: {question}")
print(f"タイプ: {type}")

question = base64.b64encode(question.encode("utf-8")).decode("ascii")

# DatabricksのワークスペースURLとトークンを設定
workspace_url = "https://adb-3805281180121377.17.azuredatabricks.net"
token = "dapi06d3afc274ef9e04877b7ace110c024d"

headers = {
    "Authorization": f"Bearer {token}"
}

# ジョブIDを指定して実行

endpoint = f"{workspace_url}/api/2.2/jobs/run-now"

if type == "1":
    payload = {
        "job_id": 991771428805335, #vector-only
        "notebook_params": {"question": question}
    }
elif type == "2":
    payload = {
        "job_id": 517140104159628, #vector-first
        "notebook_params": {"question": question}
    }
else:
    payload = {
        "job_id": 617814454526824, #graph-first
        "notebook_params": {"question": question}
    }


response = requests.post(endpoint, headers=headers, json=payload)

# API呼び出しの成功をチェック
try:
    response.raise_for_status()  # ステータスコードが200番台でない場合に例外を発生させる
except requests.exceptions.HTTPError as err:
    print(f"ジョブの実行に失敗しました: {err}")
    sys.exit(1)

print(f"Response: {response.status_code}, {response.text}")

parent_run_id=response.json()['run_id']
print(f"ジョブ実行開始。run_id: {parent_run_id}")

# 2. ジョブの完了を待つ（1分間以内、10秒ごとにチェック）
status_endpoint = f"{workspace_url}/api/2.1/jobs/runs/get?run_id={parent_run_id}"
start_time = time.time()
while True:
    status_response = requests.get(status_endpoint, headers=headers)
    status_info = status_response.json()
    state = status_info.get("state", {})
    life_cycle_state = state.get("life_cycle_state")
    print(f"現在の状態: {life_cycle_state}")

    if life_cycle_state in ["TERMINATED","SKIPPED", "INTERNAL_ERROR"]:
        break
    if time.time() - start_time > 120:  # 最大1分待機
        print("タイムアウトしました。")
        break
    time.sleep(10)

response = requests.get(f"{workspace_url}/api/2.2/jobs/runs/get?run_id={parent_run_id}", headers=headers)
run_info = response.json()
tasks = run_info.get('tasks', [])
for task in tasks:
    task_run_id = task['run_id']
    # 3. 各タスクごとに出力を取得
    output_response = requests.get(f"{workspace_url}/api/2.2/jobs/runs/get-output?run_id={task_run_id}", headers=headers)

    task_output = output_response.json().get('notebook_output', {}).get('result')
    print(f"{task['task_key']}の出力\n ", task_output)

    
