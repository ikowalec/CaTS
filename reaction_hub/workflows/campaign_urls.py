from __future__ import annotations

from django.urls import path
from .campaign_views import CampaignListView, CampaignDetailView

urlpatterns = [
    path("", CampaignListView.as_view(), name="campaign-list"),
    path("<slug:slug>/", CampaignDetailView.as_view(), name="campaign-detail"),
]
