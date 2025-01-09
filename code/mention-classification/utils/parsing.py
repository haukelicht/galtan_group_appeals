
def parse_doccano_annotation(text, annotation, keep_text: bool):
    out = {
        'start': annotation[0],
        'end': annotation[1],
        'type': annotation[2],
        'mention': text[annotation[0]:annotation[1]]
    }
    if keep_text:
        out['text'] = text
    return out
    

def unnest_sequence_annotations(data, **kwargs):
    return [
        {'text_id': line['id'], 'mention_nr': i+1} | parse_doccano_annotation(line['text'], lab, **kwargs)
        for line in data 
        for i, lab in enumerate(line['label'])
    ]