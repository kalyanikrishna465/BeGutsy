import pandas as pd
import logging

def load_metadata(metadata_file):
    try:
        # Assuming tab-delimited file; adjust delimiter if necessary
        metadata = pd.read_csv(metadata_file, sep="\t")
        logging.info(f"Loaded metadata with shape: {metadata.shape}")
        return metadata
    except Exception as e:
        logging.error(f"Error loading metadata: {e}")
        raise e

def load_taxonomy(taxonomy_file):
    try:
        # Assuming the first column is "Species" and the rest are sample IDs
        taxonomy = pd.read_csv(taxonomy_file, sep="\t")
        logging.info(f"Loaded taxonomy data with shape: {taxonomy.shape}")
        return taxonomy
    except Exception as e:
        logging.error(f"Error loading taxonomy data: {e}")
        raise e

def load_pathways(pathways_file):
    try:
        # Assuming the first column is "Pathway" and the rest are sample IDs
        pathways = pd.read_csv(pathways_file, sep="\t")
        logging.info(f"Loaded pathways data with shape: {pathways.shape}")
        return pathways
    except Exception as e:
        logging.error(f"Error loading pathways data: {e}")
        raise e

def merge_data(taxonomy, pathways, metadata):
    try:
        
        taxonomy = taxonomy.set_index(taxonomy.columns[0])
        pathways = pathways.set_index(pathways.columns[0])
        
        taxonomy_t = taxonomy.transpose()
        pathways_t = pathways.transpose()
        
        merged = pd.concat([taxonomy_t, pathways_t], axis=1, join='inner')
        logging.info(f"Merged taxonomy and pathways data shape: {merged.shape}")
        
        if 'SampleID' in metadata.columns:
            metadata = metadata.set_index("SampleID")
        merged_data = merged.join(metadata, how="inner")
        logging.info(f"Final merged data shape (with metadata): {merged_data.shape}")
        return merged_data
    except Exception as e:
        logging.error(f"Error merging data: {e}")
        raise e
