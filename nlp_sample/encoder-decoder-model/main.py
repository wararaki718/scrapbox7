from transformers import BertConfig, BertTokenizer, EncoderDecoderConfig, EncoderDecoderModel, BertTokenizerFast


def main() -> None:
    model_name = "google-bert/bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)
    print("model loaded!")

    model.save_pretrained("model/bert2bert")
    print(model.config)

    # config
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.encoder.vocab_size
    model.config.max_length = 142
    model.config.min_length = 56
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # generate
    text = (
        "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
        "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
        "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
    )
    output = tokenizer(text, return_tensors="pt")
    input_ids = output.input_ids
    print(input_ids)
    print()

    result = model.generate(input_ids)
    print(result)
    print()

    tokens = tokenizer.decode(result[0], skip_special_tokens=True)
    print(tokens)
    
    print("DONE")


if __name__ == "__main__":
    main()
