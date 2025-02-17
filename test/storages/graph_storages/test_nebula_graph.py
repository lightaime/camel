# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
import unittest
from unittest.mock import Mock, call, patch

from unstructured.documents.elements import Element

from camel.storages import NebulaGraph
from camel.storages.graph_storages.graph_element import (
    GraphElement,
    Node,
    Relationship,
)

MAX_RETRIES = 5


class TestNebulaGraph(unittest.TestCase):
    def setUp(self):
        # Mock the dependencies and external interactions
        self.host = 'localhost'
        self.username = 'user'
        self.password = 'pass'
        self.space = 'test_space'
        self.port = 9669
        self.timeout = 10000

        # Patch the methods that interact with the database
        patcher1 = patch.object(NebulaGraph, '_init_connection_pool')
        patcher2 = patch.object(NebulaGraph, '_get_session')
        self.addCleanup(patcher1.stop)
        self.addCleanup(patcher2.stop)
        self.mock_init_connection_pool = patcher1.start()
        self.mock_get_session = patcher2.start()

        # Mock the return values
        self.mock_connection_pool = Mock()
        self.mock_session = Mock()
        self.mock_init_connection_pool.return_value = self.mock_connection_pool
        self.mock_get_session.return_value = self.mock_session

        # Initialize the NebulaGraph instance with the mocks
        self.graph = NebulaGraph(
            host=self.host,
            username=self.username,
            password=self.password,
            space=self.space,
            port=self.port,
            timeout=self.timeout,
        )

    def test_query_success(self):
        # Mock session.execute to return a successful result
        mock_result_set = Mock()
        self.mock_session.execute.return_value = mock_result_set

        query_str = 'SHOW SPACES;'

        result = self.graph.query(query_str)
        self.mock_session.execute.assert_called_with(query_str)
        self.assertEqual(result, mock_result_set)

    def test_query_exception(self):
        # Mock session.execute to raise an exception
        self.mock_session.execute.side_effect = Exception('Database error')

        query_str = 'INVALID QUERY;'

        with self.assertRaises(ValueError) as context:
            self.graph.query(query_str)
        self.assertIn('Query execution error', str(context.exception))

    def test_get_relationship_types(self):
        # Mock the query method
        mock_result = Mock()
        # Mock the rows returned
        row1 = Mock()
        row1.values = [Mock()]
        row1.values[0].get_sVal.return_value = b'Relationship1'
        row2 = Mock()
        row2.values = [Mock()]
        row2.values[0].get_sVal.return_value = b'Relationship2'
        mock_result.rows.return_value = [row1, row2]
        self.graph.query = Mock(return_value=mock_result)

        rel_types = self.graph.get_relationship_types()
        self.graph.query.assert_called_with('SHOW EDGES')
        self.assertEqual(rel_types, ['Relationship1', 'Relationship2'])

    def test_add_node(self):
        node_id = 'node1'
        tag_name = 'Tag1'
        self.graph.ensure_tag_exists = Mock()
        self.graph.query = Mock()

        self.graph.add_node(node_id, tag_name)

        self.graph.ensure_tag_exists.assert_has_calls([call(tag_name, None)])
        insert_stmt = (
            f'INSERT VERTEX IF NOT EXISTS {tag_name}() VALUES "{node_id}":()'
        )
        self.graph.query.assert_called_with(insert_stmt)

    def test_add_node_with_time_label_not_in_schema(self):
        node_id = 'node1'
        tag_name = 'Tag1'
        time_label = '2025-01-21T12:00:00'

        # Mock dependencies
        self.graph.ensure_tag_exists = Mock()
        self.graph.ensure_field_in_schema = Mock()
        self.graph.query = Mock()
        self.graph._validate_time_label = Mock(return_value=time_label)

        # Mock query success
        self.graph.query.return_value.is_succeeded = Mock(return_value=True)

        # Call the method
        self.graph.add_node(node_id, tag_name, time_label)

        # Ensure the tag existence check was performed
        self.graph.ensure_tag_exists.assert_called_once_with(tag_name)

        # Ensure the time_label field was added to the schema
        self.graph.ensure_field_in_schema.assert_called_once_with(
            tag_name, "time_label", "string"
        )

        # Validate the time_label
        self.graph._validate_time_label.assert_called_once_with(time_label)

        # Ensure the correct query was executed
        insert_stmt = (
            f'INSERT VERTEX IF NOT EXISTS {tag_name}(time_label) VALUES '
            f'"{node_id}":("{time_label}")'
        )

        self.graph.query.assert_called_once_with(insert_stmt)

    def test_ensure_tag_exists_success(self):
        tag_name = 'Tag1'
        # Mock query to return a successful result
        mock_result = Mock()
        mock_result.is_succeeded.return_value = True
        self.graph.query = Mock(return_value=mock_result)

        self.graph.ensure_tag_exists(tag_name)

        create_tag_stmt = f'CREATE TAG IF NOT EXISTS {tag_name} ()'
        self.graph.query.assert_called_with(create_tag_stmt)

    @patch('time.sleep', return_value=None)
    def test_ensure_tag_exists_failure(self, mock_sleep):
        tag_name = 'Tag1'
        # Mock query to return a failed result every time
        mock_result = Mock()
        mock_result.is_succeeded.return_value = False
        mock_result.error_msg.return_value = 'Error message'
        self.graph.query = Mock(return_value=mock_result)

        with self.assertRaises(Exception) as context:
            self.graph.ensure_tag_exists(tag_name)

        self.assertIn(
            f"Failed to create tag `{tag_name}` after {MAX_RETRIES} attempts",
            str(context.exception),
        )
        self.assertEqual(self.graph.query.call_count, MAX_RETRIES)

    def test_add_triplet(self):
        subj = 'node1'
        obj = 'node2'
        rel = 'RELATESTO'
        self.graph.ensure_tag_exists = Mock()
        self.graph.ensure_edge_type_exists = Mock()
        self.graph.add_node = Mock()
        self.graph.query = Mock()

        self.graph.add_triplet(subj, obj, rel)

        self.graph.ensure_tag_exists.assert_has_calls([call(subj), call(obj)])
        self.graph.ensure_edge_type_exists.assert_has_calls([call(rel, None)])
        self.graph.add_node.assert_any_call(node_id=subj, tag_name=subj)
        self.graph.add_node.assert_any_call(node_id=obj, tag_name=obj)
        insert_stmt = (
            f'INSERT EDGE IF NOT EXISTS {rel}() VALUES "{subj}"->"{obj}":()'
        )
        self.graph.query.assert_called_with(insert_stmt)

    def test_delete_triplet(self):
        subj = 'node1'
        obj = 'node2'
        rel = 'RELATES_TO'
        self.graph.query = Mock()
        self.graph._check_edges = Mock(side_effect=[False, False])
        self.graph.delete_entity = Mock()

        self.graph.delete_triplet(subj, obj, rel)

        delete_edge_query = f'DELETE EDGE {rel} "{subj}"->"{obj}";'
        self.graph.query.assert_called_with(delete_edge_query)
        self.graph._check_edges.assert_any_call(subj)
        self.graph._check_edges.assert_any_call(obj)
        self.graph.delete_entity.assert_any_call(subj)
        self.graph.delete_entity.assert_any_call(obj)

    def test_check_edges_with_edges(self):
        entity_id = 'node1'
        mock_result = Mock()
        mock_result.is_succeeded.return_value = True
        # Mock rows with counts indicating edges exist
        row_out = Mock()
        row_out.values = [Mock()]
        row_out.values[0].get_iVal.return_value = 2
        row_in = Mock()
        row_in.values = [Mock()]
        row_in.values[0].get_iVal.return_value = 3
        mock_result.rows.return_value = [row_out, row_in]

        self.graph.query = Mock(return_value=mock_result)

        has_edges = self.graph._check_edges(entity_id)
        self.assertTrue(has_edges)
        self.graph.query.assert_called()

    def test_check_edges_no_edges(self):
        entity_id = 'node1'
        mock_result = Mock()
        mock_result.is_succeeded.return_value = True
        # Mock rows with counts indicating no edges
        row_out = Mock()
        row_out.values = [Mock()]
        row_out.values[0].get_iVal.return_value = 0
        row_in = Mock()
        row_in.values = [Mock()]
        row_in.values[0].get_iVal.return_value = 0
        mock_result.rows.return_value = [row_out, row_in]

        self.graph.query = Mock(return_value=mock_result)

        has_edges = self.graph._check_edges(entity_id)
        self.assertFalse(has_edges)
        self.graph.query.assert_called()

    def test_get_node_properties(self):
        # Mock query for 'SHOW TAGS'
        mock_show_tags_result = Mock()
        mock_show_tags_result.is_succeeded.return_value = True
        row1 = Mock()
        row1.values = [Mock()]
        row1.values[0].get_sVal.return_value = b'Tag1'
        row2 = Mock()
        row2.values = [Mock()]
        row2.values[0].get_sVal.return_value = b'Tag2'
        mock_show_tags_result.rows.return_value = [row1, row2]

        # Mock query for 'DESCRIBE TAG <tag_name>'
        mock_describe_tag_result = Mock()
        mock_describe_tag_result.is_succeeded.return_value = True
        prop_row = Mock()
        prop_row.values = [Mock()]
        prop_row.values[0].get_sVal.return_value = b'prop1'
        mock_describe_tag_result.rows.return_value = [prop_row]

        self.graph.query = Mock(
            side_effect=[
                mock_show_tags_result,
                mock_describe_tag_result,
                mock_describe_tag_result,
            ]
        )

        node_schema_props, node_structure_props = (
            self.graph.get_node_properties()
        )

        expected_node_schema_props = ['Tag1.prop1', 'Tag2.prop1']
        expected_node_structure_props = [
            {'labels': 'Tag1', 'properties': ['prop1']},
            {'labels': 'Tag2', 'properties': ['prop1']},
        ]

        self.assertEqual(node_schema_props, expected_node_schema_props)
        self.assertEqual(node_structure_props, expected_node_structure_props)

    def test_get_relationship_properties(self):
        # Mock query for 'SHOW EDGES'
        mock_show_edges_result = Mock()
        mock_show_edges_result.is_succeeded.return_value = True
        edge_row1 = Mock()
        edge_row1.values = [Mock()]
        edge_row1.values[0].get_sVal.return_value = b'Edge1'
        edge_row2 = Mock()
        edge_row2.values = [Mock()]
        edge_row2.values[0].get_sVal.return_value = b'Edge2'
        mock_show_edges_result.rows.return_value = [edge_row1, edge_row2]

        # Mock query for 'DESCRIBE EDGE <edge_name>'
        mock_describe_edge_result = Mock()
        mock_describe_edge_result.is_succeeded.return_value = True
        prop_row = Mock()
        prop_row.values = [Mock()]
        prop_row.values[0].get_sVal.return_value = b'prop1'
        mock_describe_edge_result.rows.return_value = [prop_row]

        self.graph.query = Mock(
            side_effect=[
                mock_show_edges_result,
                mock_describe_edge_result,
                mock_describe_edge_result,
            ]
        )

        rel_schema_props, rel_structure_props = (
            self.graph.get_relationship_properties()
        )

        expected_rel_schema_props = ['Edge1.prop1', 'Edge2.prop1']
        expected_rel_structure_props = [
            {'type': 'Edge1', 'properties': ['prop1']},
            {'type': 'Edge2', 'properties': ['prop1']},
        ]

        self.assertEqual(rel_schema_props, expected_rel_schema_props)
        self.assertEqual(rel_structure_props, expected_rel_structure_props)

    def test_extract_nodes(self):
        # Use actual Node instances
        node1 = Node(id='node1', type='Tag1', properties={})
        node2 = Node(id='node2', type='Tag2', properties={})
        # Create a GraphElement with nodes and optional source
        graph_elements = [
            GraphElement(
                nodes=[node1, node2],
                relationships=[],
                source=Element(element_id="a05b820b51c760a41415c57c1eef8f08"),
            )
        ]
        # Call the method
        nodes = self.graph._extract_nodes(graph_elements)
        # Expected result
        expected_nodes = [
            {'id': 'node1', 'type': 'Tag1', 'properties': {}},
            {'id': 'node2', 'type': 'Tag2', 'properties': {}},
        ]
        # Assert
        self.assertEqual(nodes, expected_nodes)

    def test_extract_relationships(self):
        # Use actual Node and Relationship instances
        node1 = Node(id='node1', type='Tag1', properties={})
        node2 = Node(id='node2', type='Tag2', properties={})
        rel = Relationship(
            subj=node1, obj=node2, type='RELATES_TO', properties={}
        )
        # Create a GraphElement with relationships and optional source
        graph_elements = [
            GraphElement(
                nodes=[],
                relationships=[rel],
                source=Element(element_id="a05b820b51c760a41415c57c1eef8f08"),
            )
        ]
        # Call the method
        relationships = self.graph._extract_relationships(graph_elements)
        # Expected result
        expected_relationships = [
            {
                'subj': {'id': 'node1', 'type': 'Tag1'},
                'obj': {'id': 'node2', 'type': 'Tag2'},
                'type': 'RELATES_TO',
            }
        ]
        # Assert
        self.assertEqual(relationships, expected_relationships)

    def test_get_indexes(self):
        # Mock query for 'SHOW TAG INDEXES'
        mock_show_indexes_result = Mock()
        mock_show_indexes_result.is_succeeded.return_value = True
        index_row = Mock()
        index_row.values = [Mock()]
        index_row.values[0].get_sVal.return_value = b'index1'
        mock_show_indexes_result.rows.return_value = [index_row]
        self.graph.query = Mock(return_value=mock_show_indexes_result)

        indexes = self.graph.get_indexes()
        expected_indexes = ['index1']
        self.assertEqual(indexes, expected_indexes)

    def test_delete_entity(self):
        entity_id = 'node1'
        self.graph.query = Mock()
        delete_vertex_query = f'DELETE VERTEX "{entity_id}";'
        self.graph.delete_entity(entity_id)
        self.graph.query.assert_called_with(delete_vertex_query)

    def test_get_schema(self):
        self.graph.get_node_properties = Mock(
            return_value=(
                ['Node.prop'],
                [{'labels': 'Node', 'properties': ['prop']}],
            )
        )
        self.graph.get_relationship_properties = Mock(
            return_value=(
                ['Rel.prop'],
                [{'type': 'Rel', 'properties': ['prop']}],
            )
        )
        self.graph.get_relationship_types = Mock(return_value=['RELATES_TO'])
        schema = self.graph.get_schema()
        expected_schema = "\n".join(
            [
                "Node properties are the following:",
                "Node.prop",
                "Relationship properties are the following:",
                "Rel.prop",
                "The relationships are the following:",
                "RELATES_TO",
            ]
        )
        self.assertEqual(schema, expected_schema)

    def test_get_structured_schema(self):
        self.graph.get_node_properties = Mock(
            return_value=(
                ['Person.name', 'Person.age'],
                [{'labels': 'Person', 'properties': ['name', 'age']}],
            )
        )
        self.graph.get_relationship_properties = Mock(
            return_value=(
                ['KNOWS.since'],
                [{'type': 'KNOWS', 'properties': ['since']}],
            )
        )
        self.graph.get_relationship_types = Mock(return_value=['KNOWS'])
        self.graph.get_indexes = Mock(return_value=[])
        structured_schema = self.graph.get_structured_schema
        expected_schema = {
            "node_props": {"Person": ["name", "age"]},
            "rel_props": {"KNOWS": ["since"]},
            "relationships": ["KNOWS"],
            "metadata": {"index": []},
        }
        self.assertEqual(structured_schema, expected_schema)

    def test_add_graph_elements(self):
        # Create actual Node and Relationship instances
        node1 = Node(id='node1', type='Tag1', properties={})
        node2 = Node(id='node2', type='Tag2', properties={})
        rel = Relationship(
            subj=node1, obj=node2, type='RELATES_TO', properties={}
        )
        # Create a GraphElement instance with nodes, relationships, and source
        graph_elements = [
            GraphElement(
                nodes=[node1, node2],
                relationships=[rel],
                source=Element(element_id="a05b820b51c760a41415c57c1eef8f08"),
            )
        ]

        # Mock the methods called within add_graph_elements
        self.graph._extract_nodes = Mock(
            return_value=[
                {'id': 'node1', 'type': 'Tag1'},
                {'id': 'node2', 'type': 'Tag2'},
            ]
        )
        self.graph._extract_relationships = Mock(
            return_value=[
                {
                    'subj': {'id': 'node1'},
                    'obj': {'id': 'node2'},
                    'type': 'RELATES_TO',
                }
            ]
        )
        self.graph.add_node = Mock()
        self.graph.add_triplet = Mock()

        # Call the method under test
        self.graph.add_graph_elements(graph_elements)
        self.graph.add_node.assert_any_call('node1', 'Tag1')
        self.graph.add_node.assert_any_call('node2', 'Tag2')
        self.graph.add_triplet.assert_called_with(
            'node1', 'node2', 'RELATES_TO'
        )

    def test_validate_time_label_valid(self):
        valid_time = "2024-12-31T21:45:22"
        result = self.graph._validate_time_label(valid_time)
        self.assertEqual(result, valid_time)

    def test_validate_time_label_none(self):
        with self.assertRaises(ValueError):
            self.graph._validate_time_label(None)

    def test_add_triplet_with_time_label(self):
        subj = 'node1'
        obj = 'node2'
        rel = 'RELATESTO'
        time_label = '2024-12-31T21:45:22'

        self.graph.ensure_tag_exists = Mock()
        self.graph.ensure_edge_type_exists = Mock()
        self.graph.add_node = Mock()
        mock_result = Mock()
        mock_result.is_succeeded.return_value = True
        self.graph.query = Mock(return_value=mock_result)

        self.graph.add_triplet(subj, obj, rel, time_label)

        self.graph.ensure_tag_exists.assert_has_calls(
            [call('node1'), call('node2')]
        )
        self.graph.ensure_edge_type_exists.assert_called_with(rel, time_label)
        self.graph.add_node.assert_any_call(node_id=subj, tag_name=subj)
        self.graph.add_node.assert_any_call(node_id=obj, tag_name=obj)

        expected_stmt = (
            f'INSERT EDGE IF NOT EXISTS {rel}(time_label) VALUES '
            f'"{subj}"->"{obj}":("{time_label}")'
        )
        self.graph.query.assert_called_with(expected_stmt)

    def test_add_triplet_with_invalid_time_label(self):
        subj = 'node1'
        obj = 'node2'
        rel = 'RELATESTO'
        invalid_time = '2024/12/31 21:45:22'  # wrong format

        with self.assertRaises(ValueError) as context:
            self.graph.add_triplet(subj, obj, rel, invalid_time)

        self.assertIn("Invalid time label format", str(context.exception))

    def test_ensure_tag_exists_with_time_label(self):
        tag_name = 'Tag1'
        time_label = '2024-12-31T21:45:22'

        mock_result = Mock()
        mock_result.is_succeeded.return_value = True
        self.graph.query = Mock(return_value=mock_result)

        self.graph.ensure_tag_exists(tag_name, time_label)

        expected_stmt = f"""CREATE TAG IF NOT EXISTS {tag_name}
            (time_label DATETIME DEFAULT {time_label})"""
        self.graph.query.assert_called_with(expected_stmt)
