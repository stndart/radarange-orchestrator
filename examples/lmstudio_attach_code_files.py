import lmstudio as lms
from radarange_orchestrator import Chat
from radarange_orchestrator.backend.lmstudio_remote_model import convert_chat

proot = '../../Work/NVmag-software'
filecontents: list[str] = Chat.prepare_code_dir(
    proot,
    include_prefix=['include/', 'src/'],
    exclude_prefix=[
        'src/Devices/spi',
        'include/Devices/spi',
        'src/Devices/uio/clock_wizard.cpp',
        'include/Devices/uio/control_regs.hpp',
        'include/Devices/uio/clock_wizard.hpp',
        'src/Server/ScpiServer.cpp',
        'src/SCPI/LMK01018.cpp',
        'src/Devices/gpio/magnetometer_gpio.cpp',
        'src/Devices/virtual/pid_generic.cpp',
        'src/Threads/temp_mon_thread.cpp',
        'src/SCPI/DAC3482.cpp',
    ],
    verbose=True,
)

client = lms.Client('95.165.10.219:1234')
model = client.llm.model('qwq-32b@q4_k_m', ttl=300)


message: str = f'<project root="{proot}">' + '\n'.join(filecontents) + '</project>'
message += 'Do you undestand, what this code does?'

total_tokens = model.count_tokens(message)
print(f'Your message is {total_tokens} tokens long')

numtokens = [model.count_tokens(content) for content in filecontents]
N = 15
for ntokens, content in sorted(zip(numtokens, filecontents), reverse=True)[:N]:
    print(ntokens, content.split('"')[1].replace('\\', '/'))

if total_tokens < model.get_context_length():
    chat = Chat()
    chat.add_user_message(message)
    for token in model.respond_stream(history=convert_chat(chat)):
        print(token.content, end='', flush=True)
print()
