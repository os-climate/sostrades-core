'''
Copyright 2025 Capgemini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import annotations

import json
import re
import time
from enum import Enum
from typing import Dict, List, Optional, Union


class DashboardAttributes(str, Enum):
    STUDY_CASE_ID = 'study_case_id'
    LAYOUT = 'layout'
    DATA = 'data'

class DisplayableItemType(str, Enum):
    TEXT = 'text'
    GRAPH = 'graph'
    SECTION = 'section'

class ItemLayout:
    def __init__(
            self,
            item_id: str,
            item_type: DisplayableItemType,
            x: int,
            y: int,
            cols: int,
            rows: int,
            minCols: int,
            minRows: int,
            children: Optional[List[str]] = None
    ):
        self.item_id = item_id
        self.item_type = item_type
        self.x = x
        self.y = y
        self.cols = cols
        self.rows = rows
        self.minCols = minCols
        self.minRows = minRows
        self.children = children if children is not None else []

    def serialize(self) -> dict:
        result = {
            "item_id": self.item_id,
            "item_type": self.item_type.value,
            "x": self.x,
            "y": self.y,
            "cols": self.cols,
            "rows": self.rows,
            "minCols": self.minCols,
            "minRows": self.minRows,
        }
        if self.children:
            result['children'] = self.children

        return result


class TextData:
    def __init__(self, content: str = ''):
        self.content = content

    def serialize(self) -> dict:
        return {
            "content": self.content
        }


class GraphData:
    def __init__(
            self,
            disciplineName: str,
            name: str,
            plotIndex: int,
            postProcessingFilters: List = None,
            graphData: Dict = None,
            title: str = None
    ):
        self.disciplineName = disciplineName
        self.name = name
        self.plotIndex = plotIndex
        self.postProcessingFilters = postProcessingFilters if postProcessingFilters is not None else []
        self.graphData = graphData if graphData is not None else {}
        self.title = title

        if not self.title and self.graphData and 'layout' in self.graphData and 'title' in self.graphData['layout']:
            self.title = re.sub(r'<[^>]*>', '', self.graphData['layout']['title']['text'])

    def serialize(self) -> dict:
        return {
            "disciplineName": self.disciplineName,
            "name": self.name,
            "plotIndex": self.plotIndex,
            "postProcessingFilters": self.postProcessingFilters,
            "graphData": self.graphData,
            "title": self.title
        }
    def id(self):
        return str({
            'disciplineName': self.disciplineName,
            'name': self.name,
            'plotIndex': self.plotIndex,
            'postProcessingFilters': self.postProcessingFilters
        })


class SectionData:
    def __init__(
            self,
            title: str = '',
            shown: bool = True,
            expandedSize: Optional[int] = None
    ):
        self.title = title
        self.shown = shown
        self.expandedSize = expandedSize

    def serialize(self) -> dict:
        result = {
            "title": self.title,
            "shown": self.shown,
        }

        if self.expandedSize is not None:
            result["expandedSize"] = self.expandedSize

        return result


class Dashboard:
    def __init__(
            self,
            study_case_id: int,
            layout: Dict[str, ItemLayout] = None,
            data: Dict[str, Union[TextData, GraphData, SectionData]] = None
    ):
        self.study_case_id = study_case_id
        self.layout = layout if layout is not None else {}
        self.data = data if data is not None else {}

    def serialize(self) -> dict:
        serialized_layout = {}
        serialized_data = {}

        for key, value in self.layout.items():
            serialized_layout[key] = value.serialize()

        for key, value in self.data.items():
            serialized_data[key] = value.serialize()

        return {
            DashboardAttributes.STUDY_CASE_ID.value: self.study_case_id,
            DashboardAttributes.LAYOUT.value: serialized_layout,
            DashboardAttributes.DATA.value: serialized_data
        }

    @classmethod
    def deserialize(cls, json_data: dict) -> Dashboard:
        """
        Create a Dashboard instance from JSON data
        Args:
            json_data (dict): JSON data representing the dashboard
        Returns:
            Dashboard: deserialized Dashboard object
        """
        structure_type = detect_dashboard_structure(json_data)
        if structure_type == 'old':
            json_data = migrate_from_old_format(json_data)

        study_case_id = json_data.get(DashboardAttributes.STUDY_CASE_ID.value)
        layout_data = json_data.get(DashboardAttributes.LAYOUT.value, {})
        data_data = json_data.get(DashboardAttributes.DATA.value, {})

        layout = {}
        data = {}

        # process layout and data
        for key, value in layout_data.items():
            item_id = value.get('item_id')
            item_type = DisplayableItemType(value.get('item_type'))
            layout[key] = ItemLayout(
                item_id=item_id,
                item_type=item_type,
                x=int(value.get('x', 0)),
                y=int(value.get('y', 0)),
                cols=int(value.get('cols', 1)),
                rows=int(value.get('rows', 1)),
                minCols=int(value.get('minCols', 1)),
                minRows=int(value.get('minRows', 1))
            )
            if 'children' in value:
                layout[key].children = value['children']
            if key in data_data:
                item_data = data_data[key]
                if item_type == DisplayableItemType.TEXT:
                    data[key] = TextData(content=item_data.get('content', ''))
                elif item_type == DisplayableItemType.GRAPH:
                    data[key] = GraphData(
                        disciplineName=item_data.get('disciplineName', ''),
                        name=item_data.get('name', ''),
                        plotIndex=int(item_data.get('plotIndex', 0)),
                        postProcessingFilters=item_data.get('postProcessingFilters', []),
                        graphData=item_data.get('graphData', {}),
                        title=item_data.get('title')
                    )
                elif item_type == DisplayableItemType.SECTION:
                    data[key] = SectionData(
                        title=item_data.get('title', ''),
                        shown=item_data.get('shown', True),
                        expandedSize=item_data.get('expandedSize')
                    )

        for data_key in data_data.keys():
            if data_key not in layout:
                item_data = data_data[data_key]
                if 'content' in item_data:
                    data[data_key] = TextData(content=item_data.get('content', ''))
                else:
                    data[data_key] = GraphData(
                        disciplineName=item_data.get('disciplineName', ''),
                        name=item_data.get('name', ''),
                        plotIndex=int(item_data.get('plotIndex', 0)),
                        postProcessingFilters=item_data.get('postProcessingFilters', []),
                        graphData=item_data.get('graphData', {}),
                        title=item_data.get('title')
                    )

        return cls(study_case_id=study_case_id, layout=layout, data=data)

def detect_dashboard_structure(json_data: dict) -> str:
    """Detect if dashboard uses old or new structure"""
    if 'items' in json_data and isinstance(json_data['items'], List):
        return 'old'
    elif 'layout' in json_data and 'data' in json_data and isinstance(json_data['layout'], dict) and isinstance(json_data['data'], dict):
        return 'new'
    else:
        return 'unknown'

def migrate_from_old_format(dashboard_json):
    """
    Migrate old dashboard data format to the new format.
    This method assumes that the old format has a specific structure that needs to be transformed.
    """
    old_data = json.loads(dashboard_json) if isinstance(dashboard_json, str) else dashboard_json

    new_dashboard = {
        'study_case_id': old_data.get(DashboardAttributes.STUDY_CASE_ID.value, 0),
        'layout': {},
        'data': {}
    }
    old_items = old_data.get('items', [])

    for old_item in old_items:
        item_type = old_item.get('type')
        if item_type == DisplayableItemType.GRAPH:
            # not possible to migrate graph item -> lacking filters to create the new item_id
            continue
        layout, data = migrate_item_by_type(old_item)
        item_id = layout['item_id']
        new_dashboard['layout'][item_id] = layout
        new_dashboard['data'][item_id] = data
        if item_type == DisplayableItemType.SECTION:
            old_section_items = old_item.get('data', {}).get('items', [])
            for child_item in old_section_items:
                if item_type == DisplayableItemType.GRAPH:
                    # not possible to migrate graph item -> lacking filters to create the new item_id
                    continue
                child_layout, child_data = migrate_item_by_type(child_item)
                new_dashboard['data'][child_item.get('id')] = child_data

    return new_dashboard

def migrate_item_by_type(old_item):
    """Helper function to migrate item base on its type"""
    item_type = old_item.get('type')
    if item_type == DisplayableItemType.TEXT:
        return migrate_text_item(old_item)
    elif item_type == DisplayableItemType.SECTION:
        return migrate_section_item(old_item)
    else:
        raise ValueError(f"Unknown item type: {item_type}")

def migrate_text_item(old_item):
    """Migrate a text item from the old format to the new format."""
    item_id = old_item.get('id', f"text_{int(time.time() * 1000)}")
    layout = {
        'item_id': item_id,
        'item_type': DisplayableItemType.TEXT.value,
        'x': old_item.get('x', 0),
        'y': old_item.get('y', 0),
        'cols': old_item.get('cols', 12),
        'rows': old_item.get('rows', 8),
        'minCols': old_item.get('minCols', 1),
        'minRows': old_item.get('minRows', 1),
    }

    data = {
        'content': old_item.get('data', {}).get('content', '')
    }

    return layout, data

def migrate_section_item(old_item):
    """Migrate a section item from the old format to the new format."""
    item_id = old_item.get('id', f"section_{int(time.time() * 1000)}")
    children_ids = []
    old_section_items = old_item.get('data', {}).get('items', [])
    for child_item in old_section_items:
        if child_item.get('type') == DisplayableItemType.GRAPH:
            # not possible to migrate graph item -> lacking filters to create the new item_id
            continue
        child_id = child_item.get('id')
        if child_id:
            children_ids.append(child_id)

    layout = {
        'item_id': item_id,
        'item_type': DisplayableItemType.SECTION.value,
        'x': old_item.get('x', 0),
        'y': old_item.get('y', 0),
        'cols': old_item.get('cols', 40),
        'rows': old_item.get('rows', 20),
        'minCols': old_item.get('minCols', 40),
        'minRows': old_item.get('minRows', 16),
        'children': children_ids
    }

    old_section_data = old_item.get('data', {})
    data = {
        'title': old_section_data.get('title', ''),
        'shown': old_section_data.get('shown', True),
        'expandedSize': old_section_data.get('expandedSize')
    }

    return layout, data
