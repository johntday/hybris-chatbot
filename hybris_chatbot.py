import os
import streamlit as st
from dotenv import load_dotenv
from mods.history import ChatHistory
from mods.layout import Layout
from mods.utils import Utilities
from mods.sidebar import Sidebar


# To be able to update the changes made to modules in localhost,
# you can press the "r" key on the localhost page to refresh and reflect the changes made to the module files.
def reload_module(module_name):
    import importlib
    import sys
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    return sys.modules[module_name]


history_module = reload_module('mods.history')
layout_module = reload_module('mods.layout')
utils_module = reload_module('mods.utils')
sidebar_module = reload_module('mods.sidebar')

ChatHistory = history_module.ChatHistory
Layout = layout_module.Layout
Utilities = utils_module.Utilities
Sidebar = sidebar_module.Sidebar

uploaded_file = {'name': 'Hybris', 'type': 'text/csv', 'size': 0}


# def init():
#     load_dotenv()
#     st.set_page_config(layout="wide", page_icon="💬", page_title="Hybris ChatBot")
#     user_api_key = Utilities.load_api_key()
#
#     if not st.session_state.get("chatbot"):
#         st.session_state["model"] = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
#         st.session_state["temperature"] = 0.0
#         chatbot = Utilities.setup_chatbot(
#             uploaded_file, st.session_state["model"], st.session_state["temperature"]
#         )
#         st.session_state["ready"] = True
#         st.session_state["chatbot"] = chatbot


def main():
    app_name = os.getenv("APP_NAME", "Hybris ChatBot")
    st.set_page_config(layout="wide", page_icon="💬", page_title="Hybris ChatBot")
    load_dotenv()
    layout, sidebar, utils = Layout(), Sidebar(), Utilities()
    layout.show_header()

    user_api_key = Utilities.load_api_key()

    if not user_api_key:
        layout.show_api_key_missing()
    else:
        os.environ["OPENAI_API_KEY"] = user_api_key

        if not st.session_state.get("chatbot"):
            st.session_state["model"] = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
            st.session_state["temperature"] = 0.0
            chatbot = Utilities.setup_chatbot(
                uploaded_file, st.session_state["model"], st.session_state["temperature"]
            )
            st.session_state["ready"] = True
            st.session_state["chatbot"] = chatbot

        # uploaded_file = utils.handle_upload()

        if uploaded_file:
            history = ChatHistory()
            sidebar.show_options()

            # uploaded_file_content = BytesIO(uploaded_file.getvalue())
            uploaded_file_content = 'a'

            try:
                if st.session_state["ready"]:
                    response_container, prompt_container = st.container(), st.container()

                    with prompt_container:
                        is_ready, user_input = layout.prompt_form()

                        history.initialize(uploaded_file)
                        if st.session_state["reset_chat"]:
                            history.reset(uploaded_file)

                        if is_ready:
                            history.append("user", user_input)
                            output = st.session_state["chatbot"].conversational_chat(user_input)
                            history.append("assistant", output)
                            sidebar.show_sources(st.session_state["chat_sources"])

                    history.generate_messages(response_container)

                    # if st.session_state["show_csv_agent"] and uploaded_file_content != 'a':
                    #     query = st.text_input(label="Use CSV agent for precise information about the structure of your csv file",
                    #                           placeholder="ex : how many rows in my file ?")
                    #     if query != "":
                    #         old_stdout = sys.stdout
                    #         sys.stdout = captured_output = StringIO()
                    #         agent = create_csv_agent(ChatOpenAI(temperature=0), uploaded_file_content, verbose=True, max_iterations=4)
                    #
                    #         result = agent.run(query)
                    #
                    #         sys.stdout = old_stdout
                    #         thoughts = captured_output.getvalue()
                    #
                    #         cleaned_thoughts = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', thoughts)
                    #         cleaned_thoughts = re.sub(r'\[1m>', '', cleaned_thoughts)
                    #
                    #         with st.expander("Afficher les pensées de l'agent"):
                    #             st.write(cleaned_thoughts)
                    #
                    #         st.write(result)

            except Exception as e:
                st.error(f"Error: {str(e)}")

    sidebar.about(app_name)


if __name__ == "__main__":
    main()
