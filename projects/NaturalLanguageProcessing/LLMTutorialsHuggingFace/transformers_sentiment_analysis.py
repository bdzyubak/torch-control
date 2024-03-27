# Based on: https://huggingface.co/learn/nlp-course/chapter1/3?fw=pt

from transformers import pipeline


def main():
    classifier_distilbert = pipeline("sentiment-analysis")  # TODO: add model spec. Default is distilbert
    result = classifier_distilbert("I've been waiting for a HuggingFace course my whole life.")

    # The output is a list with each entry being a dictionary of label, confidence for each input list member
    assert result[0]['label'] == 'POSITIVE', 'Failed to detect positive sentiment'

    print(result[0]['score'])

    result = classifier_distilbert(["I've been waiting for a HuggingFace course my whole life.",
                                    "I hate this so much!"])

    assert result[0]['label'] == 'POSITIVE' and result[1]['label'] == 'NEGATIVE', 'Failed to distinguish sentiments'

    text = ["I am so sure about this.", "This is complicated", "This is complex", "This is challenging",
            "Are you sure about this?", "I am not so sure about this.", "I am not sure about this."]
    classify_text_print_results(text, classifier_distilbert)
    # Cool!

    # Let's make this harder
    print()
    print('DISTILBERT')
    text = ["I like talking to myself", "There are voices in my head", "I am always so happy",
            "I am never sad", "I multitask a lot", "I like bad things.", 'Distilbert is a hallucination',
            "So good it's bad", "So bad it's good"]

    classify_text_print_results(text, classifier_distilbert)
    # "Voices in my head" is positive with 82% confidence. As long as they're helpful muses, it's okay. John Nash
    # had them.
    # So good it's bad is positive, as is the least obvious always happy/never sad.

    ## Let's try a few other models
    # Has untrained layers. Requires fine-tuning, otherwise appends text to input phrases.
    print()
    print('roberta-base-go_emotions')
    # classifier_sam_lowe = pipeline("text-generation", model="SamLowe/roberta-base-go_emotions")
    # classify_text_print_results(text, classifier_sam_lowe)

    print()
    print('roberta-base-go_emotions')
    classifier_sam_lowe = pipeline("text-generation", model="ProsusAI/finbert")
    classify_text_print_results(text, classifier_sam_lowe)


def classify_text_print_results(text, classifier):
    predictions = classifier(text)
    for ind, sentence in enumerate(text):
        print(f"{sentence} - {predictions[ind]['label']} - {predictions[ind]['score']}")


if __name__ == '__main__':
    main()
