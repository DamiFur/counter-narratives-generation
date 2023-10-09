from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
import string
import argparse


parser = argparse.ArgumentParser(description="Train models for identifying argumentative components inside the ASFOCONG dataset")

parser.add_argument("model_name", type=str, choices=["google/flan-t5-base", "mistralai/Mistral-7B-Instruct-v0.1", "tiiuae/falcon-7b-instruct"])
parser.add_argument("--hs", type=str, default="despertó la alcaldesa de la ciudad en la que un inmigrante con papeles casi fue muerto por inmigrantes sin papeles al defender a una mujer hostigada por éstos ¿no tendrá nada que decir? ¿le parecerá bien que los manteros hostiguen a señoras y peguen a turistas?")
parser.add_argument("--collective", type=str, default="manteros")
parser.add_argument("--property", type=str, default="hostiguen a señoras y peguen a turistas")
parser.add_argument("--justification", type=str, default="despertó la alcaldesa de la ciudad en la que un inmigrante con papeles casi fue muerto por inmigrantes sin papeles al defender a una mujer hostigada por éstos ¿no tendrá nada que decir?")
parser.add_argument("--conclusion", type=str, default="¿le parecerá bien que los manteros hostiguen a señoras y peguen a turistas?")
args = parser.parse_args()

model_name = args.model_name
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

from transformers import StoppingCriteria, StoppingCriteriaList

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if tokenizer.decode(stop) == tokenizer.decode(last_token):
                return True
        return False

if model_name == "tiiuae/falcon-7b-instruct":
    stop_words = ["]", "']", "']\n", "\n", "]\n", "\n\n", "']\n\n", "<|endoftext|>"]
else:
    stop_words = ["]", "']", "']\n", "\n", "]\n", "\n\n", "']\n\n", "</s>"]
stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

if model_name == "tiiuae/falcon-7b-instruct":
    prompt_noinfo = f"Genera una respuesta contraria a los siguientes mensajes de odio:\n'¿esto podría explicar el brutismo caribeño y el subsahariano? ¿habrá correlación? extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings' -> Respuesta: 'Es difícil que algo tenga relación con algún estudio de internet completamente descontextualizado...'\n 'llegan 900 venezolanos por día al perú cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ? qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema solo reaccionarán cuando esto se desborde' -> Respuesta: 'Estás siendo paranoico. No se va a desbordar, y es por eso que no hace falta ponerle coto y fin'\n '{args.hs}' -> Respuesta: "
    prompt_collective = f"Genera una respuesta contraria a los siguientes mensajes de odio:\n'¿esto podría explicar el brutismo caribeño y el subsahariano? ¿habrá correlación? extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings' | Collectivo contra quien se dirige el odio: 'caribeño y subsahariano' | Propiedad asociada al colectivo: 'brutismo' -> Respuesta: 'No es cierto que los caribeños o subsaharianos sean brutos. Es solo un prejuicio.'\n 'llegan 900 venezolanos por día al perú cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ? qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema solo reaccionarán cuando esto se desborde' | Collectivo contra quien se dirige el odio: 'venezolanos' | Propiedad asociada al colectivo: 'invasion' -> Respuesta: 'Lo llamas invasión pero solo te refieres a los venezolanos, y no a los demás inmigrantes. ¿No estás siendo un poco racista?'\n '{args.hs}' | Collectivo contra quien se dirige el odio: '{args.collective}' | Propiedad asociada al colectivo: '{args.property}' -> Respuesta: "
    prompt_premises = f"Genera una respuesta contraria a los siguientes mensajes de odio:\n'¿esto podría explicar el brutismo caribeño y el subsahariano? ¿habrá correlación? extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings' | Justificación del argumento: extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings  | Conclusión: ¿esto podría explicar el brutismo caribeño y el subsahariano ? ¿habrá correlación?' -> Respuesta: 'Es difícil que algo tenga relación con algún estudio de internet completamente descontextualizado...'\n 'llegan 900 venezolanos por día al perú cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ? qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema solo reaccionarán cuando esto se desborde' | Justificación del argumento: 'llegan 900 venezolanos por día al perú' y 'qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema' | Conclusión: 'cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ?' y 'solo reaccionarán cuando esto se desborde'  -> Respuesta: 'Estás siendo paranoico. No se va a desbordar, y es por eso que no hace falta ponerle coto y fin'\n '{args.hs}' | Justificación del argumento: '{args.justification}' | Conclusión: '{args.conclusion}'  -> Respuesta: "
else:
    prompt_noinfo = [
        {"role": "user", "content": "Genera una respuesta contraria a los siguientes mensajes de odio:\n'¿esto podría explicar el brutismo caribeño y el subsahariano? ¿habrá correlación? extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings"},
        {"role": "assistant", "content": "Es difícil que algo tenga relación con algún estudio de internet completamente descontextualizado..."},
        {"role": "user", "content": "llegan 900 venezolanos por día al perú cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ? qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema solo reaccionarán cuando esto se desborde"},
        {"role": "assistant", "content": "Estás siendo paranoico. No se va a desbordar, y es por eso que no hace falta ponerle coto y fin"},
        {"role": "user", "content": f"{args.hs}"}
    ]
    prompt_collective = [
        {"role": "user", "content": "Genera una respuesta contraria a los siguientes mensajes de odio:\n'¿esto podría explicar el brutismo caribeño y el subsahariano? ¿habrá correlación? extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings' | Collectivo contra quien se dirige el odio: 'caribeño y subsahariano' | Propiedad asociada al colectivo: 'brutismo'"},
        {"role": "assistant", "content": "No es cierto que los caribeños o subsaharianos sean brutos. Es solo un prejuicio."},
        {"role": "user", "content": "llegan 900 venezolanos por día al perú cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ? qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema solo reaccionarán cuando esto se desborde' | Collectivo contra quien se dirige el odio: 'venezolanos' | Propiedad asociada al colectivo: 'invasion'"},
        {"role": "assistant", "content": "Lo llamas invasión pero solo te refieres a los venezolanos, y no a los demás inmigrantes. ¿No estás siendo un poco racista?"},
        {"role": "user", "content": f"{args.hs} | Collectivo contra quien se dirige el odio: '{args.collective}' | Propiedad asociada al colectivo: '{args.property}'"}
    ]
    prompt_premises = [
        {"role": "user", "content": "Genera una respuesta contraria a los siguientes mensajes de odio:\n'¿esto podría explicar el brutismo caribeño y el subsahariano? ¿habrá correlación? extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings' | Justificación del argumento: extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings  | Conclusión: ¿esto podría explicar el brutismo caribeño y el subsahariano ? ¿habrá correlación?"},
        {"role": "assistant", "content": "Es difícil que algo tenga relación con algún estudio de internet completamente descontextualizado..."},
        {"role": "user", "content": "llegan 900 venezolanos por día al perú cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ? qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema solo reaccionarán cuando esto se desborde' | Justificación del argumento: 'llegan 900 venezolanos por día al perú' y 'qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema' | Conclusión: 'cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ?' y 'solo reaccionarán cuando esto se desborde'"},
        {"role": "assistant", "content": "Estás siendo paranoico. No se va a desbordar, y es por eso que no hace falta ponerle coto y fin"},
        {"role": "user", "content": f"{args.hs} | Justificación del argumento: {args.justification} | Conclusión: {args.conclusion}"}
    ]


def generate_answers(prompt, num_samples=10):
  # define some source text and tokenize it
  source_text = prompt
  if model_name == "tiiuae/falcon-7b-instruct":
    source_ids = tokenizer(source_text, return_tensors="pt").input_ids.to("cuda")
  else:
      source_ids = tokenizer.apply_chat_template(prompt, return_tensors="pt").to("cuda")

  gen_outputs = []
  for _ in range(num_samples):
    # generate the output using beam search
    gen_output = model.generate(
        inputs=source_ids,
        # temperature=temperature,
        do_sample=True,
        max_new_tokens=40,
        num_beams=4,
        no_repeat_ngram_size=2,
        num_return_sequences=1, # only show top beams
        # early_stopping=True,
        stopping_criteria=stopping_criteria,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_outputs.append(gen_output)

  return gen_outputs

outputs_noinfo = generate_answers(prompt_noinfo)
print("NoInfo")
print(tokenizer.batch_decode(outputs_noinfo[0])[0])
print("---------------------")
print("Collective")
outputs_noinfo = generate_answers(prompt_collective)
print(tokenizer.batch_decode(outputs_noinfo[0])[0])
print("---------------------")
print("Premises")
outputs_noinfo = generate_answers(prompt_premises)
print(tokenizer.batch_decode(outputs_noinfo[0])[0])
print("---------------------")