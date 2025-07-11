import lmstudio as lms

with lms.Client('192.168.1.10:1234') as client:
    # model = client.llm.model('gemma-3-27b-it-qat', ttl=30)
    model = client.llm.model('gemma-3-27b-it-abliterated', ttl=30)

    image_path = 'examples/downloads/image.jpg'
    image_handle = client.prepare_image(image_path)

    chat = lms.Chat()
    chat.add_user_message('Что на картинке?', images=[image_handle])

    for token in model.respond_stream(chat):
        print(token.content, end='', flush=True)
    print()
