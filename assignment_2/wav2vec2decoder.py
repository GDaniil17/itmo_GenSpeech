from typing import List, Tuple, Union

import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class Wav2Vec2Decoder:
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-960h",
            lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
            beam_width=3,
            alpha=1.0,
            beta=1.0
        ):
        """
        Initialization of Wav2Vec2Decoder class

        Args:
            model_name (str): Pretrained Wav2Vec2 model from transformers
            lm_model_path (str): Path to the KenLM n-gram model (for LM rescoring)
            beam_width (int): Number of hypotheses to keep in beam search
            alpha (float): LM weight for shallow fusion and rescoring
            beta (float): Word bonus for shallow fusion
        """
        # once logits are available, no other interactions with the model are allowed
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        # you can interact with these parameters
        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path)

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V)

        Returns:
            str: Decoded transcript
        """
        indices = torch.argmax(logits, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i.item() for i in indices if i != self.blank_token_id]
        joined = "".join([self.vocab[i] for i in indices])
        return joined.replace(self.word_delimiter, " ").strip()

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        """
        Perform beam search decoding (no LM)

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size
            return_beams (bool): Return all beam hypotheses for second pass LM rescoring

        Returns:
            Union[str, List[Tuple[List[int], float]]]:
                (str) - If return_beams is False, returns the best decoded transcript as a string.
                (List[Tuple[List[int], float]]) - If return_beams is True, returns a list of tuples
                    containing hypotheses and log probabilities.
        """
        log_probs = torch.log_softmax(logits, dim=-1)

        beams = [([], 0.0)]  # (prefix, log_prob)

        for t in range(len(log_probs)):
            new_beams = {}

            for prefix, prefix_score in beams:
                blank_prob = log_probs[t][self.blank_token_id].item()
                new_prefix = prefix.copy()  # No change to prefix
                new_score = prefix_score + blank_prob

                prefix_key = tuple(new_prefix)
                if prefix_key not in new_beams or new_score > new_beams[prefix_key]:
                    new_beams[prefix_key] = new_score

                for token in range(len(log_probs[t])):
                    if token == self.blank_token_id:
                        continue

                    token_prob = log_probs[t][token].item()

                    if prefix and prefix[-1] == token:
                        new_prefix = prefix.copy()
                        new_score = prefix_score + token_prob

                        prefix_key = tuple(new_prefix)
                        if prefix_key not in new_beams or new_score > new_beams[prefix_key]:
                            new_beams[prefix_key] = new_score

                    new_prefix = prefix.copy()
                    new_prefix.append(token)
                    new_score = prefix_score + token_prob

                    prefix_key = tuple(new_prefix)
                    if prefix_key not in new_beams or new_score > new_beams[prefix_key]:
                        new_beams[prefix_key] = new_score

            beams = []
            for prefix, score in sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:self.beam_width]:
                beams.append((list(prefix), score))

        if return_beams:
            return beams
        else:
            best_beam = max(beams, key=lambda x: x[1])
            best_tokens = best_beam[0]

            result = ""
            for token in best_tokens:
                if self.vocab[token] == self.word_delimiter:
                    result += ' '
                else:
                    result += self.vocab[token]

            return result.strip()

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        """
        Perform beam search decoding with shallow LM fusion

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size

        Returns:
            str: Decoded transcript
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM shallow fusion")

        T, V = logits.size()
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        beams = [([], 0.0, "<s>")]

        for t in range(T):
            all_candidates = []
            for seq, score, lm_state in beams:
                for v in range(V):
                    if v == self.blank_token_id:
                        new_seq = seq.copy()
                        logp = log_probs[t, v].item()
                        all_candidates.append((new_seq, score + logp, lm_state))
                        continue
                    
                    token = self.vocab[v]
                    new_seq = seq + [v]
                    logp = log_probs[t, v].item()

                    if token == self.word_delimiter:
                        text = "".join(self.vocab[i] for i in seq).replace(self.word_delimiter, " ")
                        words = text.strip().split()
                        last_word = words[-1] if words else ""
                        lm_score = self.lm_model.score(last_word, bos=False, eos=False)
                        total_score = score + logp + self.alpha * lm_score + self.beta
                        all_candidates.append((new_seq, total_score, "<s>"))
                    else:
                        total_score = score + logp
                        all_candidates.append((new_seq, total_score, lm_state))

            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:self.beam_width]

        if not beams:
            return ""
            
        best_seq = beams[0][0]
        if not best_seq:
            return ""
            
        best_seq_tensor = torch.tensor(best_seq, dtype=torch.long)

        filtered = torch.unique_consecutive(best_seq_tensor, dim=-1)
        filtered = [i.item() for i in filtered if i.item() != self.blank_token_id]
        decoded = "".join([self.vocab[i] for i in filtered]).replace(self.word_delimiter, " ").strip()

        return decoded

    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs

        Args:
            beams (list): List of tuples (hypothesis, log_prob)

        Returns:
            str: Best rescored transcript
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")

        rescored_beams = []

        for tokens, acoustic_score in beams:
            if not tokens:
                continue

            transcript = ""
            for token in tokens:
                if self.vocab[token] == self.word_delimiter:
                    transcript += ' '
                else:
                    transcript += self.vocab[token]

            if len(transcript.strip()) < 2:
                continue

            lm_score = self.alpha * self.lm_model.score(transcript)
            
            word_count = len(transcript.split())
            word_bonus = self.beta * word_count

            final_score = acoustic_score + lm_score + word_bonus

            rescored_beams.append((transcript.strip(), final_score))

        rescored_beams.sort(key=lambda x: x[1], reverse=True)

        if rescored_beams:
            return rescored_beams[0][0]
        else:
            return ""

    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Decode input audio file using the specified method

        Args:
            audio_input (torch.Tensor): Audio tensor
            method (str): Decoding method ("greedy", "beam", "beam_lm", "beam_lm_rescore"),
                where "greedy" is a greedy decoding,
                      "beam" is beam search without LM,
                      "beam_lm" is beam search with LM shallow fusion, and
                      "beam_lm_rescore" is a beam search with second pass LM rescoring

        Returns:
            str: Decoded transcription
        """
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError("Invalid decoding method. Choose one of 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'.")


def test(decoder, audio_path, true_transcription):

    import Levenshtein

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, "Audio sample rate must be 16kHz"

    print("=" * 60)
    print("Target transcription")
    print(true_transcription)

    # Print all decoding methods results
    for d_strategy in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        print("-" * 60)
        print(f"{d_strategy} decoding")
        transcript = decoder.decode(audio_input, method=d_strategy)
        print(f"{transcript}")
        print(f"Character-level Levenshtein distance: {Levenshtein.distance(true_transcription, transcript.strip())}")


if __name__ == "__main__":

    test_samples = [
        ("examples/sample1.wav", "IF YOU ARE GENEROUS HERE IS A FITTING OPPORTUNITY FOR THE EXERCISE OF YOUR MAGNANIMITY IF YOU ARE PROUD HERE AM I YOUR RIVAL READY TO ACKNOWLEDGE MYSELF YOUR DEBTOR FOR AN ACT OF THE MOST NOBLE FORBEARANCE"),
        ("examples/sample2.wav", "AND IF ANY OF THE OTHER COPS HAD PRIVATE RACKETS OF THEIR OWN IZZY WAS UNDOUBTEDLY THE MAN TO FIND IT OUT AND USE THE INFORMATION WITH A BEAT SUCH AS THAT EVEN GOING HALVES AND WITH ALL THE GRAFT TO THE UPPER BRACKETS HE'D STILL BE ABLE TO MAKE HIS PILE IN A MATTER OF MONTHS"),
        ("examples/sample3.wav", "GUESS A MAN GETS USED TO ANYTHING HELL MAYBE I CAN HIRE SOME BUMS TO SIT AROUND AND WHOOP IT UP WHEN THE SHIPS COME IN AND BILL THIS AS A REAL OLD MARTIAN DEN OF SIN"),
        ("examples/sample4.wav", "IT WAS A TUNE THEY HAD ALL HEARD HUNDREDS OF TIMES SO THERE WAS NO DIFFICULTY IN TURNING OUT A PASSABLE IMITATION OF IT TO THE IMPROVISED STRAINS OF I DIDN'T WANT TO DO IT THE PRISONER STRODE FORTH TO FREEDOM"),
        ("examples/sample5.wav", "MARGUERITE TIRED OUT WITH THIS LONG CONFESSION THREW HERSELF BACK ON THE SOFA AND TO STIFLE A SLIGHT COUGH PUT UP HER HANDKERCHIEF TO HER LIPS AND FROM THAT TO HER EYES"),
        ("examples/sample6.wav", "AT THIS TIME ALL PARTICIPANTS ARE IN A LISTEN ONLY MODE"),
        ("examples/sample7.wav", "THE INCREASE WAS MAINLY ATTRIBUTABLE TO THE NET INCREASE IN THE AVERAGE SIZE OF OUR FLEETS"),
        ("examples/sample8.wav", "OPERATING SURPLUS IS A NON CAP FINANCIAL MEASURE WHICH IS DEFINED AS FULLY IN OUR PRESS RELEASE"),
    ]

    decoder = Wav2Vec2Decoder()

    _ = [test(decoder, audio_path, target) for audio_path, target in test_samples]
