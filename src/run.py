import os
import uvicorn

def main():
    os.environ["PYTHONPATH"] = "src"
    uvicorn.run("cp_genie.main:app", reload=True)