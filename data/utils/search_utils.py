from typing import Optional, Union
from langchain.docstore.document import Document
import wikipedia

class LangChainWiki:

    def __init__(self) -> None:
        self.document: Optional[Document] = None
        self.lookup_str = ""
        self.lookup_index = 0

    def search(self, search: str) -> Union[str, Document]:
        def _try_search(term: str) -> Union[str, Document]:
            try:
                page_content = wikipedia.page(search).content 
                url = wikipedia.page(search).url 
                result: Union[str, Document] = Document( page_content=page_content, metadata={"page": url} )
                return result
            except wikipedia.PageError:
                return f"Could not find [{term}]. Similar: {wikipedia.search(term)}"
            except wikipedia.DisambiguationError:
                return f"Could not find [{term}]. Similar: {wikipedia.search(term)}"
            except Exception:
                return f"Could not find [{term}]. Similar: {wikipedia.search(term)}"

        result = _try_search(search)

        if isinstance(result, str) and "Similar:" in result:
            try:
                similar = wikipedia.search(search)
                if similar:
                    fallback = similar[0]
                    print(f"[INFO] Falling back to similar term: {fallback}")
                    result = _try_search(fallback)
            except Exception as e:
                print(f"[ERROR] Could not fetch similar terms: {e}")

        if isinstance(result, Document):
            self.document = result
            return self._sumary
        else:
            self.document = None
            return result

    def lookup(self, term: str):
        if self.document is None:
            raise ValueError("Cannot lookup without a successful search first")
        if term.lower() != self.lookup_str:
            self.lookup_str = term.lower()
            self.lookup_index = 0
        else:
            self.lookup_index += 1
        lookups = [p for p in self._paragraphs if self.lookup_str in p.lower()]
        if len(lookups) == 0:
            return "No Results"
        elif self.lookup_index >= len(lookups):
            return "No More Results"
        else:
            result_prefix = f"(Result {self.lookup_index + 1}/{len(lookups)})"
            return f"{result_prefix} {lookups[self.lookup_index]}"

    @property
    def _sumary(self) -> str:
        return self._paragraphs[0]

    @property
    def _paragraphs(self) -> list[str]:
        if self.document is None:
            raise ValueError("Cannot get paragraphs without a document")
        return self.document.page_content.split("\n\n")
