from __future__ import annotations


class FriendlyTablePaginationMixin:
    page_size_default = 25
    page_size_options = (10, 25, 50, 100)

    def _requested_page_size(self) -> int:
        raw = str(self.request.GET.get("page_size") or "").strip()
        if not raw:
            return int(self.page_size_default)
        try:
            parsed = int(raw)
        except ValueError:
            return int(self.page_size_default)
        valid = tuple(int(value) for value in self.page_size_options)
        return parsed if parsed in valid else int(self.page_size_default)

    def get_table_pagination(self, table):  # noqa: ARG002
        return {"per_page": self._requested_page_size()}

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["page_size"] = self._requested_page_size()
        context["page_size_options"] = tuple(int(value) for value in self.page_size_options)
        return context
