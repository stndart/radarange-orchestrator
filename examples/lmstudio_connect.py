import lmstudio as lms

with lms.Client("95.165.10.219:1234") as client:
    model = client.llm.model("qwq-32b@q4_k_m", ttl=30)
    print(model.respond("How's things?"))