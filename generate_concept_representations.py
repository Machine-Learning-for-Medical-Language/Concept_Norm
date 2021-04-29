import read_files as read


def n2c2_input():
    ontology = read.read_from_tsv(
        "data/n2c2/processed/input_joint/umls/train.tsv")
    train = read.read_from_tsv(
        "data/n2c2/processed/input_joint/mention/train.tsv")
    dev = read.read_from_tsv("data/n2c2/processed/input_joint/mention/dev.tsv")

    ontology = ontology + train + dev

    cuis = read.read_from_json(
        "data/n2c2/triplet_network/st_subpool/ontology_cui")
    norm_mention = {}
    for idx, [_, norm, mention] in enumerate(ontology):
        read.add_dict(norm_mention, norm, mention)

    mentions = []
    idx = 0
    cui_mention_idx = {}

    concept_synonyms = {}

    for cui in cuis:
        cui_mentions = list(set(norm_mention[cui]))
        concept_synonyms[cui] = cui_mentions
        mentions += cui_mentions
        if len(cui_mentions) == 0:
            print(idx)
        end = idx + len(cui_mentions)
        cui_mention_idx[cui] = (idx, end)
        idx = end
    input_mentions = [[syn] for syn in mentions]

    read.save_in_json(
        "data/n2c2/triplet_network/con_norm/ontology_concept_synonyms",
        concept_synonyms)

    print(len(cui_mention_idx))

    read.save_in_tsv(
        "data/n2c2/triplet_network/con_norm/ontology_synonyms.tsv",
        input_mentions)
    read.save_in_json(
        "data/n2c2/triplet_network/con_norm/ontology_concept_synonyms_idx",
        cui_mention_idx)

    return mentions, cui_mention_idx


n2c2_input()
