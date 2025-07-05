import lmstudio as lms

with lms.Client("95.165.10.219:1234") as client:
    for c in client.list_downloaded_models():
        print(c.model_key, c.info.display_name)