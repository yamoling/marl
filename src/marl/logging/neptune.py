import neptune
from dotenv import load_dotenv
import os

load_dotenv()
run = neptune.init_run(
    project=f"{os.getenv('NEPTUNE_WORKSPACE')}/{os.getenv('NEPTUNE_PROJECT')}",
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
    name=os.getlogin(),
)


x = run["accuracy"]
