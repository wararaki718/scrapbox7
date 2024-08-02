from transformers import BertConfig, BertTokenizer, EncoderDecoderConfig, EncoderDecoderModel


def main() -> None:
    model_name = "google-bert/bert-base-uncased"

    encoder_config = BertConfig()
    decoder_config = BertConfig()

    config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    model = EncoderDecoderModel(config)

    encoder_config = model.config.encoder
    decoder_config = model.config.decoder

    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True

    model_path = "model/bert2bert"
    model.save_pretrained(model_path)
    print("model saved!")

    encoder_decoder_config = EncoderDecoderConfig.from_pretrained(model_path)
    # model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_path, config=encoder_decoder_config)
    model = EncoderDecoderModel.from_pretrained(model_path)
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.encoder.vocab_size

    text = (
        "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
        "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
        "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
    )
    output = tokenizer(text, return_tensors="pt")
    # print(output)
    input_ids = output.input_ids

    generated_ids = model.generate(input_ids)
    print(generated_ids)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)

    print("DONE")


if __name__ == "__main__":
    main()
