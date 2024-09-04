from __future__ import annotations
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from ylz_utils.langchain import LangchainLib
    
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate,SystemMessagePromptTemplate
from langchain_core.prompt_values import StringPromptValue,ChatPromptValue
from langchain_core.pydantic_v1 import BaseModel

from ylz_utils.data import StringLib

class PromptLib():
    def get_prompt(self,system_prompt=None,human_keys={"input":"input"},
                   outputParser:Optional[BaseModel]=None,
                   history_messages_key="history",
                   use_chat = False,
                   use_chinese=True) -> ChatPromptTemplate:
            if not system_prompt:
                system_prompt=""
            if use_chinese:
                system_prompt = f"所有问题请用中文回答\n{system_prompt}"
            if not use_chat:
                human_input_keys = []
                human_prompt = ""
                if human_keys:
                    for key in human_keys:
                        human_prompt += f"{human_keys[key]}:{{{key}}}\n" 
                    human_input_keys = human_keys.keys()
                if outputParser:
                    prompt = PromptTemplate(
                        template=f"{system_prompt}\n{{format_instructions}}\n{human_prompt}",
                        input_variables=human_input_keys,
                        partial_variables={"format_instructions": outputParser.get_format_instructions()}
                    )
                else:
                    prompt =  PromptTemplate(
                        template=f"{system_prompt}\n{human_prompt}",
                        input_variables=human_input_keys,
                    ) 
            else:
                messages = []
                if outputParser:
                    partial_prompt = PromptTemplate(template=f"{system_prompt}\n{{format_instructions}}", 
                                                    partial_variables={"format_instructions": outputParser.get_format_instructions()})
                    #messages.append(("system",partial_prompt.format(**{"format_instructions": outputParser.get_format_instructions() })))
                    messages.append(
                        SystemMessagePromptTemplate(prompt=partial_prompt)               
                    )
                elif system_prompt:
                    messages.append(("system",system_prompt))

                messages.append(("placeholder", f"{{{history_messages_key}}}"))
                human_prompt = ""
                if human_keys:
                    for key in human_keys:
                        human_prompt += f"{human_keys[key]}:{{{key}}}\n"
                    messages.append(("human",human_prompt))
                StringLib.logging_in_box(str(messages),console_width=160,print_func=print)
                prompt = ChatPromptTemplate.from_messages(messages)
            return prompt
