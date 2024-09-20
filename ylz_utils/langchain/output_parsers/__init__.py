from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ylz_utils.langchain import LangchainLib
    
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.output_parsers import RetryOutputParser

from langchain_core.prompts import PromptTemplate

from pydantic import BaseModel, Field
from typing import List,Literal

class OutputParserLib():
   def get_outputParser(self,pydantic_object=None,fix=False,llm=None,retry=1):
        NAIVE_FIX = """Instructions:
            --------------
            {instructions}
            --------------
            Completion:
            --------------
            {input}
            --------------

            Above, the Completion did not satisfy the constraints given in the Instructions.
            Error:
            --------------
            {error}
            --------------

            Please try again. Please only respond with an answer that satisfies the constraints laid out in the Instructions:"""

        PROMPT = PromptTemplate.from_template(NAIVE_FIX)
        
        if pydantic_object:
            parser =  PydanticOutputParser(pydantic_object=pydantic_object)
        else:
            parser = StrOutputParser()
        if fix:
            if not llm:
                llm = self.get_llm()
            OutputFixingParser.legacy = False
            parser =  OutputFixingParser.from_llm(
                llm = llm,
                prompt = PROMPT ,
                parser = parser,
                max_retries = retry
            )
        return parser 