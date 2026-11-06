"""Compatibility identifiers for previously trained native checkpoints."""
CONFIG_SCHEMA = 'mindchen_native_config_v1'
SUPPORTED_CONFIG_SCHEMAS = (CONFIG_SCHEMA, 'minimind_native_config_v1')
MODEL_TYPES = {'minimind': 'mindchen', 'minimind-v': 'mindchen-v'}
LEGACY_PROJECT_DIR_NAMES = ('minimind-master', 'minimind-v-master', 'mindchen-master', 'mindchen')


def normalize_model_config(values):
    result = dict(values)
    if 'model_type' in result:
        result['model_type'] = MODEL_TYPES.get(result['model_type'], result['model_type'])
    return result
