"""Клиент для записи, чтения, поиска и удаления векторов в Pinecone."""

from __future__ import annotations

import os
from typing import Mapping, NotRequired, Sequence, TypedDict

from dotenv import load_dotenv
from pinecone import DenseVectorQuery, EmbedQuery, Index, Pinecone

# Имя переменной с ключом доступа к Pinecone.
ENV_API_KEY = "PINECONE_API_KEY"
# Имя переменной с названием уже созданного индекса.
ENV_INDEX_NAME = "PINECONE_INDEX"
# Сколько ближайших векторов вернуть, если вызывающий не указал top_k.
DEFAULT_TOP_K = 5
# Верхняя граница top_k в API Pinecone.
MAX_TOP_K = 10000
# Пустая строка — пространство имён по умолчанию у векторного индекса.
DEFAULT_NAMESPACE = ""
# Документный индекс не принимает пустое пространство имён.
DEFAULT_DOCUMENT_NAMESPACE = "default"

# Допустимое значение метаданных: строка, число, флаг или список строк.
TMetadataValue = str | int | float | bool | list[str]
# Набор метаданных, который Pinecone умеет фильтровать.
TMetadata = dict[str, TMetadataValue]


class TVectorRecord(TypedDict):
    """Вектор или документ, который нужно записать в индекс."""

    id: str
    values: NotRequired[list[float]]
    metadata: NotRequired[TMetadata]


class TStoredVector(TypedDict):
    """Вектор, прочитанный из индекса по идентификатору."""

    id: str
    values: list[float]
    metadata: TMetadata


class TSearchMatch(TypedDict):
    """Один результат поиска по близости."""

    id: str
    score: float
    values: NotRequired[list[float]]
    metadata: NotRequired[TMetadata]


class TConnectionInfo(TypedDict):
    """Сведения об открытом подключении к индексу."""

    index_name: str
    host: str


class PineconeClient:
    """Клиент векторного индекса Pinecone.

    Входные значения конструктора:
        api_key: ключ API. Если не передан, читается из ``PINECONE_API_KEY``.
        index_name: имя уже созданного индекса. Если не передано, читается
            из ``PINECONE_INDEX``.
        namespace: пространство имён по умолчанию для всех операций.
            Пустая строка означает пространство индекса по умолчанию.

    Выходные значения:
        Экземпляр клиента. Сетевое подключение появляется только после
        вызова ``connect``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
        namespace: str = DEFAULT_NAMESPACE,
    ) -> None:
        # Подхватываем .env, чтобы ключ не лежал в исходниках.
        load_dotenv()
        self._api_key = (api_key or os.getenv(ENV_API_KEY) or "").strip()
        self._index_name = (index_name or os.getenv(ENV_INDEX_NAME) or "").strip()
        self._namespace = namespace
        self._client: Pinecone | None = None
        self._index: Index | None = None
        # Индекс со схемой документов не принимает сырые векторы.
        self._uses_documents = False
        self._text_field: str | None = None
        self._vector_field: str | None = None

    def connect(self) -> TConnectionInfo:
        """Открывает клиент Pinecone и подключается к индексу.

        Входные значения:
            Нет. Используются ключ и имя индекса, заданные в конструкторе
            или в переменных окружения.

        Выходные значения:
            TConnectionInfo: имя индекса и адрес его data plane.

        Raises:
            ValueError: ключ или имя индекса пустые.
        """
        if not self._api_key:
            raise ValueError(
                "Не задан API-ключ Pinecone. Передайте api_key "
                "или установите PINECONE_API_KEY."
            )
        if not self._index_name:
            raise ValueError(
                "Не задано имя индекса. Передайте index_name "
                "или установите PINECONE_INDEX."
            )

        self._client = Pinecone(api_key=self._api_key)
        self._load_schema()
        self._index = self._client.index(name=self._index_name)
        return {
            "index_name": self._index_name,
            "host": self._index.host,
        }

    def write(
        self,
        vectors: Sequence[TVectorRecord],
        namespace: str | None = None,
    ) -> int:
        """Записывает векторы в индекс. Существующий id перезаписывается.

        Входные значения:
            vectors: записи с полями ``id``, ``values`` и необязательным
                ``metadata``.
            namespace: пространство имён. Если не передано, берётся значение
                из конструктора.

        Выходные значения:
            int: сколько векторов принял Pinecone.

        Raises:
            RuntimeError: ``connect`` ещё не вызывался.
            ValueError: список векторов пуст или запись заполнена неверно.
        """
        index = self._require_index()
        if not vectors:
            raise ValueError("Список векторов для записи пуст.")

        target_namespace = self._resolve_namespace(namespace)
        if self._uses_documents:
            documents = [self._to_document(vector) for vector in vectors]
            response = index.documents.upsert(
                namespace=target_namespace,
                documents=documents,
            )
            return int(response.upserted_count)

        payload = [_to_upsert_item(vector) for vector in vectors]
        vector_response = index.upsert(
            vectors=payload,
            namespace=target_namespace,
            show_progress=False,
        )
        return int(vector_response.upserted_count)

    def read(
        self,
        ids: Sequence[str],
        namespace: str | None = None,
    ) -> dict[str, TStoredVector]:
        """Читает векторы по идентификаторам, без ранжирования.

        Входные значения:
            ids: идентификаторы векторов. Список не должен быть пустым.
            namespace: пространство имён. Если не передано, берётся значение
                из конструктора.

        Выходные значения:
            dict[str, TStoredVector]: найденные векторы по id. Идентификатор,
            которого нет в пространстве имён, в словарь не попадает.

        Raises:
            RuntimeError: ``connect`` ещё не вызывался.
            ValueError: список идентификаторов пуст.
        """
        index = self._require_index()
        if not ids:
            raise ValueError("Список идентификаторов для чтения пуст.")

        target_namespace = self._resolve_namespace(namespace)
        if self._uses_documents:
            response = index.documents.fetch(
                namespace=target_namespace,
                ids=list(ids),
            )
            stored: dict[str, TStoredVector] = {}
            for document in response.documents.values():
                stored[document.id] = self._document_to_stored(document.to_dict())
            return stored

        vector_response = index.fetch(
            ids=list(ids),
            namespace=target_namespace,
        )
        stored = {}
        for vector_id, vector in vector_response.vectors.items():
            stored[vector_id] = {
                "id": vector.id,
                "values": list(vector.values),
                "metadata": _copy_metadata(vector.metadata),
            }
        return stored

    def search(
        self,
        vector: Sequence[float] | None = None,
        top_k: int = DEFAULT_TOP_K,
        namespace: str | None = None,
        metadata_filter: Mapping[str, object] | None = None,
        include_values: bool = False,
        include_metadata: bool = True,
        query_text: str | None = None,
    ) -> list[TSearchMatch]:
        """Ищет ближайшие векторы к переданному.

        Входные значения:
            vector: вектор запроса той же размерности, что и поле dense_vector.
            query_text: текст запроса, если индекс сам считает эмбеддинг.
            top_k: сколько ближайших результатов вернуть, от 1 до 10000.
            namespace: пространство имён. Если не передано, берётся значение
                из конструктора.
            metadata_filter: фильтр метаданных Pinecone, например
                ``{"lang": {"$eq": "ru"}}``.
            include_values: вернуть ли координаты найденных векторов.
            include_metadata: вернуть ли метаданные найденных векторов.

        Выходные значения:
            list[TSearchMatch]: совпадения от большего score к меньшему.
            У каждого элемента есть ``id`` и ``score``. Поля ``values`` и
            ``metadata`` присутствуют, только если их попросили вернуть.

        Raises:
            RuntimeError: ``connect`` ещё не вызывался.
            ValueError: вектор запроса пуст или top_k вне диапазона.
        """
        index = self._require_index()
        if top_k < 1 or top_k > MAX_TOP_K:
            raise ValueError(f"top_k должен быть в диапазоне от 1 до {MAX_TOP_K}.")

        target_namespace = self._resolve_namespace(namespace)
        query_filter = dict(metadata_filter) if metadata_filter is not None else None
        if self._uses_documents:
            return self._search_documents(
                index=index,
                vector=vector,
                query_text=query_text,
                top_k=top_k,
                namespace=target_namespace,
                metadata_filter=query_filter,
                include_values=include_values,
                include_metadata=include_metadata,
            )
        if not vector:
            raise ValueError("Вектор запроса пуст.")

        response = index.query(
            vector=[float(value) for value in vector],
            top_k=top_k,
            namespace=target_namespace,
            filter=query_filter,
            include_values=include_values,
            include_metadata=include_metadata,
        )
        matches: list[TSearchMatch] = []
        for match in response.matches:
            item: TSearchMatch = {
                "id": match.id,
                "score": float(match.score),
            }
            if include_values:
                item["values"] = list(match.values)
            if include_metadata:
                item["metadata"] = _copy_metadata(match.metadata)
            matches.append(item)
        return matches

    def delete(
        self,
        ids: Sequence[str] | None = None,
        namespace: str | None = None,
        metadata_filter: Mapping[str, object] | None = None,
        delete_all: bool = False,
    ) -> None:
        """Удаляет векторы по id, по фильтру или все в пространстве имён.

        Входные значения:
            ids: идентификаторы для удаления.
            namespace: пространство имён. Если не передано, берётся значение
                из конструктора.
            metadata_filter: фильтр метаданных. Удаляются все векторы,
                которые ему соответствуют.
            delete_all: если True, очищает всё пространство имён.
                Нужен ровно один способ выбора: ids, фильтр или delete_all.

        Выходные значения:
            None. Успешный ответ Pinecone не содержит тела.

        Raises:
            RuntimeError: ``connect`` ещё не вызывался.
            ValueError: не выбран ровно один способ удаления.
        """
        index = self._require_index()
        selected = (
            ids is not None,
            metadata_filter is not None,
            delete_all,
        )
        if sum(selected) != 1:
            raise ValueError(
                "Укажите ровно один способ удаления: ids, metadata_filter "
                "или delete_all=True."
            )
        if ids is not None and len(ids) == 0:
            raise ValueError("Список идентификаторов для удаления пуст.")
        if metadata_filter is not None and len(metadata_filter) == 0:
            raise ValueError("Фильтр для удаления пуст.")

        target_namespace = self._resolve_namespace(namespace)
        query_filter = dict(metadata_filter) if metadata_filter is not None else None
        if self._uses_documents:
            index.documents.delete(
                namespace=target_namespace,
                ids=list(ids) if ids is not None else None,
                delete_all=delete_all,
                filter=query_filter,
            )
            return

        index.delete(
            ids=list(ids) if ids is not None else None,
            delete_all=delete_all,
            filter=query_filter,
            namespace=target_namespace,
        )

    def _require_index(self) -> Index:
        """Возвращает открытый индекс или сообщает, что подключения нет."""
        if self._index is None:
            raise RuntimeError("Сначала вызовите connect().")
        return self._index

    def _resolve_namespace(self, namespace: str | None) -> str:
        """Выбирает пространство имён вызова или значение по умолчанию."""
        chosen = self._namespace if namespace is None else namespace
        if chosen == "" and self._uses_documents:
            return DEFAULT_DOCUMENT_NAMESPACE
        return chosen

    def _load_schema(self) -> None:
        """Читает схему индекса и запоминает поле текста или вектора.

        Входные значения:
            Нет. Нужен уже созданный клиент Pinecone.

        Выходные значения:
            None. Результат сохраняется в полях экземпляра.
        """
        if self._client is None:
            raise RuntimeError("Сначала создайте клиент Pinecone.")

        description = self._client.indexes.describe(self._index_name)
        schema = description.schema
        raw_fields: dict[str, object] = {}
        if schema is not None:
            raw = schema.to_dict()
            fields = raw.get("fields", {})
            if isinstance(fields, dict):
                raw_fields = fields

        self._uses_documents = len(raw_fields) > 0
        self._text_field = None
        self._vector_field = None
        for name, spec in raw_fields.items():
            if not isinstance(spec, dict):
                continue
            if spec.get("type") == "dense_vector" and self._vector_field is None:
                self._vector_field = str(name)
            if spec.get("embed") and self._text_field is None:
                self._text_field = str(name)

    def _to_document(self, vector: TVectorRecord) -> dict[str, object]:
        """Собирает документ для индекса со схемой.

        Входные значения:
            vector: запись с ``id``, необязательным ``values`` и ``metadata``.

        Выходные значения:
            dict[str, object]: документ с ключом ``_id`` и полями схемы.
        """
        vector_id = str(vector["id"]).strip()
        if vector_id == "":
            raise ValueError("У документа пустой id.")

        metadata = dict(vector.get("metadata") or {})
        document: dict[str, object] = {"_id": vector_id}
        if self._vector_field is not None:
            values = vector.get("values") or []
            if len(values) == 0:
                raise ValueError(f"У документа {vector_id} пустой вектор.")
            document[self._vector_field] = [float(value) for value in values]
        if self._text_field is not None:
            phrase = metadata.get(self._text_field)
            if not isinstance(phrase, str) or phrase.strip() == "":
                phrase = metadata.get("phrase")
            if not isinstance(phrase, str) or phrase.strip() == "":
                raise ValueError(
                    f"Индекс ждёт текст в поле {self._text_field}. "
                    "Положите фразу в metadata['phrase']."
                )
            document[self._text_field] = phrase

        for key, value in metadata.items():
            if key in document:
                continue
            normalized = _normalize_metadata_value(value)
            if normalized is not None:
                document[key] = normalized
        return document

    def _document_to_stored(self, raw: Mapping[str, object]) -> TStoredVector:
        """Переводит документ Pinecone в запись чтения.

        Входные значения:
            raw: словарь документа, включая служебные ключи.

        Выходные значения:
            TStoredVector: id, координаты и остальные поля как метаданные.
        """
        vector_id = str(raw.get("_id", ""))
        values: list[float] = []
        if self._vector_field is not None:
            raw_values = raw.get(self._vector_field)
            if isinstance(raw_values, list):
                values = [float(value) for value in raw_values]

        metadata: TMetadata = {}
        for key, value in raw.items():
            if key.startswith("_") or key == self._vector_field:
                continue
            normalized = _normalize_metadata_value(value)
            if normalized is not None:
                metadata[str(key)] = normalized
        return {"id": vector_id, "values": values, "metadata": metadata}

    def _search_documents(
        self,
        index: Index,
        vector: Sequence[float] | None,
        query_text: str | None,
        top_k: int,
        namespace: str,
        metadata_filter: dict[str, object] | None,
        include_values: bool,
        include_metadata: bool,
    ) -> list[TSearchMatch]:
        """Ищет документы по тексту или по готовому вектору.

        Входные значения:
            index: открытый индекс.
            vector: готовый вектор, если в схеме есть поле dense_vector.
            query_text: текст запроса для поля с встроенным эмбеддингом.
            top_k: число результатов.
            namespace: непустое пространство имён.
            metadata_filter: фильтр полей документа.
            include_values: вернуть ли координаты.
            include_metadata: вернуть ли остальные поля.

        Выходные значения:
            list[TSearchMatch]: совпадения от большего score к меньшему.
        """
        if self._text_field is not None and query_text:
            score_by = [EmbedQuery(field=self._text_field, query=query_text)]
        elif self._vector_field is not None and vector:
            score_by = [
                DenseVectorQuery(
                    field=self._vector_field,
                    values=[float(value) for value in vector],
                )
            ]
        elif self._text_field is not None:
            raise ValueError(
                f"Этот индекс сам считает эмбеддинг поля {self._text_field}. "
                "Передайте текст запроса в query_text."
            )
        else:
            raise ValueError("Для поиска нужен вектор или текст запроса.")

        include_fields = ["*"] if include_metadata or include_values else None
        response = index.documents.search(
            namespace=namespace,
            score_by=score_by,
            top_k=top_k,
            include_fields=include_fields,
            filter=metadata_filter,
        )
        matches: list[TSearchMatch] = []
        for document in response.matches:
            stored = self._document_to_stored(document.to_dict())
            item: TSearchMatch = {
                "id": document.id,
                "score": float(document.score or 0.0),
            }
            if include_values:
                item["values"] = stored["values"]
            if include_metadata:
                item["metadata"] = stored["metadata"]
            matches.append(item)
        return matches


def _to_upsert_item(vector: TVectorRecord) -> dict[str, object]:
    """Проверяет запись и приводит её к виду, который ждёт upsert.

    Входные значения:
        vector: запись с ``id``, ``values`` и необязательным ``metadata``.

    Выходные значения:
        dict[str, object]: словарь для ``Index.upsert``.
    """
    vector_id = str(vector["id"]).strip()
    values = vector["values"]
    if vector_id == "":
        raise ValueError("У вектора пустой id.")
    if len(values) == 0:
        raise ValueError(f"У вектора {vector_id} пустой список values.")

    item: dict[str, object] = {
        "id": vector_id,
        "values": [float(value) for value in values],
    }
    metadata = vector.get("metadata")
    if metadata:
        item["metadata"] = dict(metadata)
    return item


def _copy_metadata(raw: Mapping[str, object] | None) -> TMetadata:
    """Оставляет в метаданных только типы, которые принимает Pinecone.

    Входные значения:
        raw: метаданные из ответа Pinecone или None.

    Выходные значения:
        TMetadata: копия с допустимыми значениями. Пустой словарь, если
        метаданных не было.
    """
    if not raw:
        return {}

    copied: TMetadata = {}
    for key, value in raw.items():
        normalized = _normalize_metadata_value(value)
        if normalized is not None:
            copied[key] = normalized
    return copied


def _normalize_metadata_value(value: object) -> TMetadataValue | None:
    """Приводит одно значение метаданных к допустимому типу.

    Входные значения:
        value: произвольное значение из ответа Pinecone.

    Выходные значения:
        TMetadataValue, если тип подходит, иначе None.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [item for item in value if isinstance(item, str)]
    return None
