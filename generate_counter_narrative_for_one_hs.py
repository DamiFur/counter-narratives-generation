from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
import string

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
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

stop_words = ["]", "']", "']\n", "\n", "]\n", "\n\n", "']\n\n", "<|endoftext|>"]
stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

prompt_noinfo = "Genera una respuesta contraria a los siguientes mensajes de odio:\n'¿esto podría explicar el brutismo caribeño y el subsahariano? ¿habrá correlación? extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings' -> Respuesta: 'Es difícil que algo tenga relación con algún estudio de internet completamente descontextualizado...'\n 'llegan 900 venezolanos por día al perú cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ? qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema solo reaccionarán cuando esto se desborde' -> Respuesta: 'Estás siendo paranoico. No se va a desbordar, y es por eso que no hace falta ponerle coto y fin'\n 'despertó la alcaldesa de la ciudad en la que un inmigrante con papeles casi fue muerto por inmigrantes sin papeles al defender a una mujer hostigada por éstos ¿no tendrá nada que decir? ¿le parecerá bien que los manteros hostiguen a señoras y peguen a turistas?' -> Respuesta: "
prompt_collective = "Genera una respuesta contraria a los siguientes mensajes de odio:\n'¿esto podría explicar el brutismo caribeño y el subsahariano? ¿habrá correlación? extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings' | Collectivo contra quien se dirige el odio: 'caribeño y subsahariano' | Propiedad asociada al colectivo: 'brutismo' -> Respuesta: 'No es cierto que los caribeños o subsaharianos sean brutos. Es solo un prejuicio.'\n 'llegan 900 venezolanos por día al perú cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ? qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema solo reaccionarán cuando esto se desborde' | Collectivo contra quien se dirige el odio: 'venezolanos' | Propiedad asociada al colectivo: 'invasion' -> Respuesta: 'Lo llamas invasión pero solo te refieres a los venezolanos, y no a los demás inmigrantes. ¿No estás siendo un poco racista?'\n 'despertó la alcaldesa de la ciudad en la que un inmigrante con papeles casi fue muerto por inmigrantes sin papeles al defender a una mujer hostigada por éstos ¿no tendrá nada que decir? ¿le parecerá bien que los manteros hostiguen a señoras y peguen a turistas?' | Collectivo contra quien se dirige el odio: 'manteros' | Propiedad asociada al colectivo: 'hostiguen a señoras y peguen a turistas' -> Respuesta: "
prompt_premises = "Genera una respuesta contraria a los siguientes mensajes de odio:\n'¿esto podría explicar el brutismo caribeño y el subsahariano? ¿habrá correlación? extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings' | Justificación del argumento: extreme heat linked with reduced cognitive performance among young adults in non-air-conditioned buildings  | Conclusión: ¿esto podría explicar el brutismo caribeño y el subsahariano ? ¿habrá correlación?  -> Respuesta: 'Es difícil que algo tenga relación con algún estudio de internet completamente descontextualizado...'\n 'llegan 900 venezolanos por día al perú cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ? qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema solo reaccionarán cuando esto se desborde' | Justificación del argumento: 'llegan 900 venezolanos por día al perú' y 'qué ya no es una inmigración a las autoridades peruanas no les da ni la tos por este tema' | Conclusión: 'cuantos más tendrán que venir para que ya se ponga coto y fin a esta invasión ?' y 'solo reaccionarán cuando esto se desborde'  -> Respuesta: 'Estás siendo paranoico. No se va a desbordar, y es por eso que no hace falta ponerle coto y fin'\n 'despertó la alcaldesa de la ciudad en la que un inmigrante con papeles casi fue muerto por inmigrantes sin papeles al defender a una mujer hostigada por éstos ¿no tendrá nada que decir? ¿le parecerá bien que los manteros hostiguen a señoras y peguen a turistas?' | Justificación del argumento: 'despertó la alcaldesa de la ciudad en la que un inmigrante con papeles casi fue muerto por inmigrantes sin papeles al defender a una mujer hostigada por éstos ¿no tendrá nada que decir?' | Conclusión: '¿le parecerá bien que los manteros hostiguen a señoras y peguen a turistas?'  -> Respuesta: "

def generate_answers(prompt, model, num_samples=10):
  # define some source text and tokenize it
  source_text = prompt
  source_ids = tokenizer(source_text, return_tensors="pt").input_ids.to("cuda")

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
print(tokenizer.batch_decode(outputs_noinfo[0])[0].replace(prompt_noinfo, ""))
print("---------------------")
print("Collective")
outputs_noinfo = generate_answers(prompt_collective)
print(tokenizer.batch_decode(outputs_noinfo[0])[0].replace(prompt_collective, ""))
print("---------------------")
print("Premises")
outputs_noinfo = generate_answers(prompt_premises)
print(tokenizer.batch_decode(outputs_noinfo[0])[0].replace(prompt_premises, ""))
print("---------------------")