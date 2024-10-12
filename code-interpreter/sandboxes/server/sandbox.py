from fastapi import FastAPI
import uvicorn
import subprocess
import nest_asyncio

nest_asyncio.apply()
sandbox = FastAPI()


@sandbox.post("/sandboxes/voltscript")
async def run_code(request: dict):
    try:
        prompt = request['query']
        with open('input.vss', 'w') as file:
            file.write(prompt)
        write_cmd = ['docker', 'cp', 'input.vss', 'voltscript-sandbox:workspace/input.vss']
        _ = subprocess.run(write_cmd, capture_output=True, text=True)
        if _.stderr:
            raise Exception((_.stderr))
        run_cmd = ['docker', 'exec', 'voltscript-sandbox', 'VoltScript', 'input.vss']
        res = subprocess.run(run_cmd, capture_output=True, text=True)
        if res.stderr:
            raise Exception(str(res.stderr))
        response = res.stdout
    except Exception as e:
        response = f"Error: {str(e)}"
    return {'response': response, "subprocesses": [str(_), str(res)], "prompt": prompt}

@sandbox.post("/sandboxes/javascript")
async def run_code(request: dict):
    try:
        prompt = request['query']
        with open('input.js', 'w') as file:
            file.write(prompt)
        write_cmd = ['docker', 'cp', 'input.js', 'javascript-sandbox:workspace/input.js']
        _ = subprocess.run(write_cmd, capture_output=True, text=True)
        if _.stderr:
            raise Exception((_.stderr))
        run_cmd = ['docker', 'exec', 'javascript-sandbox', 'node', 'input.js']
        res = subprocess.run(run_cmd, capture_output=True, text=True)
        if res.stderr:
            raise Exception(str(res.stderr))
        response = res.stdout
    except Exception as e:
        response = f"Error: {str(e)}"
    return {'response': response, "subprocesses": [str(_), str(res)], "prompt": prompt}

@sandbox.post("/sandboxes/python")
async def run_code(request: dict):
    try:
        prompt = request['query']
        with open('input.py', 'w') as file:
            file.write(prompt)
        write_cmd = ['docker', 'cp', 'input.py', 'python-sandbox:workspace/input.py']
        _ = subprocess.run(write_cmd, capture_output=True, text=True)
        if _.stderr:
            raise Exception((_.stderr))
        run_cmd = ['docker', 'exec', 'python-sandbox', 'python', 'input.py']
        res = subprocess.run(run_cmd, capture_output=True, text=True)
        if res.stderr:
            raise Exception(str(res.stderr))
        response = res.stdout
    except Exception as e:
        response = f"Error: {str(e)}"
    return {'response': response, "subprocesses": [str(_), str(res)], "prompt": prompt}

if __name__ == "__main__":
    uvicorn.run(sandbox, host="0.0.0.0", port=8081)