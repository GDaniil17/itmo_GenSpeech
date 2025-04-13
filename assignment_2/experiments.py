import os

import Levenshtein
import matplotlib.pyplot as plt
import pandas as pd
import torchaudio
pandoc asr_decoding_report.md -o asr_decoding_report.pdf --pdf-engine=xelatex -V geometry:margin=1in
from wav2vec2decoder import Wav2Vec2Decoder
markdown-pdf asr_decoding_report.md

def word_error_rate(reference, hypothesis):
    """
    Calculate Word Error Rate between reference and hypothesis
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    distance = Levenshtein.distance(ref_words, hyp_words)

    # Calculate WER
    return distance / len(ref_words) if len(ref_words) > 0 else 0


def run_decoding_experiments(lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz"):
    """
    Run experiments with different decoding methods and report metrics
    """
    test_samples = [
        ("examples/sample1.wav",
         "IF YOU ARE GENEROUS HERE IS A FITTING OPPORTUNITY FOR THE EXERCISE OF YOUR MAGNANIMITY IF YOU ARE PROUD HERE AM I YOUR RIVAL READY TO ACKNOWLEDGE MYSELF YOUR DEBTOR FOR AN ACT OF THE MOST NOBLE FORBEARANCE"),
        ("examples/sample2.wav",
         "AND IF ANY OF THE OTHER COPS HAD PRIVATE RACKETS OF THEIR OWN IZZY WAS UNDOUBTEDLY THE MAN TO FIND IT OUT AND USE THE INFORMATION WITH A BEAT SUCH AS THAT EVEN GOING HALVES AND WITH ALL THE GRAFT TO THE UPPER BRACKETS HE'D STILL BE ABLE TO MAKE HIS PILE IN A MATTER OF MONTHS"),
        ("examples/sample3.wav",
         "GUESS A MAN GETS USED TO ANYTHING HELL MAYBE I CAN HIRE SOME BUMS TO SIT AROUND AND WHOOP IT UP WHEN THE SHIPS COME IN AND BILL THIS AS A REAL OLD MARTIAN DEN OF SIN"),
        ("examples/sample4.wav",
         "IT WAS A TUNE THEY HAD ALL HEARD HUNDREDS OF TIMES SO THERE WAS NO DIFFICULTY IN TURNING OUT A PASSABLE IMITATION OF IT TO THE IMPROVISED STRAINS OF I DIDN'T WANT TO DO IT THE PRISONER STRODE FORTH TO FREEDOM"),
        ("examples/sample5.wav",
         "MARGUERITE TIRED OUT WITH THIS LONG CONFESSION THREW HERSELF BACK ON THE SOFA AND TO STIFLE A SLIGHT COUGH PUT UP HER HANDKERCHIEF TO HER LIPS AND FROM THAT TO HER EYES"),
        ("examples/sample6.wav", "AT THIS TIME ALL PARTICIPANTS ARE IN A LISTEN ONLY MODE"),
        ("examples/sample7.wav",
         "THE INCREASE WAS MAINLY ATTRIBUTABLE TO THE NET INCREASE IN THE AVERAGE SIZE OF OUR FLEETS"),
        ("examples/sample8.wav",
         "OPERATING SURPLUS IS A NON CAP FINANCIAL MEASURE WHICH IS DEFINED AS FULLY IN OUR PRESS RELEASE"),
    ]

    decoder = Wav2Vec2Decoder(lm_model_path=lm_model_path)

    methods = ["greedy", "beam", "beam_lm", "beam_lm_rescore"]

    results = []

    for audio_path, target in test_samples:
        audio_input, sr = torchaudio.load(audio_path)
        assert sr == 16000, "Audio sample rate must be 16kHz"

        sample_name = os.path.basename(audio_path)

        for method in methods:
            transcript = decoder.decode(audio_input, method=method)

            cer = Levenshtein.distance(target, transcript) / len(target)
            wer = word_error_rate(target, transcript)

            results.append({
                "Sample": sample_name,
                "Method": method,
                "Transcript": transcript,
                "CER": cer,
                "WER": wer
            })

    results_df = pd.DataFrame(results)

    summary = results_df.groupby("Method")[["CER", "WER"]].mean().reset_index()

    print("\n===== DECODING METHODS COMPARISON =====")
    print(summary)

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.bar(summary["Method"], summary["CER"])
    plt.title("Character Error Rate by Decoding Method")
    plt.ylabel("CER")
    plt.ylim(0, 1)

    plt.subplot(1, 2, 2)
    plt.bar(summary["Method"], summary["WER"])
    plt.title("Word Error Rate by Decoding Method")
    plt.ylabel("WER")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig("decoding_methods_comparison.png")

    return results_df, summary


def experiment_beam_width(beam_widths=[1, 3, 5, 10], lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz"):
    """
    Experiment with different beam widths
    """
    test_samples = [
        ("examples/sample1.wav",
         "IF YOU ARE GENEROUS HERE IS A FITTING OPPORTUNITY FOR THE EXERCISE OF YOUR MAGNANIMITY IF YOU ARE PROUD HERE AM I YOUR RIVAL READY TO ACKNOWLEDGE MYSELF YOUR DEBTOR FOR AN ACT OF THE MOST NOBLE FORBEARANCE"),
        ("examples/sample3.wav",
         "GUESS A MAN GETS USED TO ANYTHING HELL MAYBE I CAN HIRE SOME BUMS TO SIT AROUND AND WHOOP IT UP WHEN THE SHIPS COME IN AND BILL THIS AS A REAL OLD MARTIAN DEN OF SIN"),
        ("examples/sample6.wav", "AT THIS TIME ALL PARTICIPANTS ARE IN A LISTEN ONLY MODE"),
    ]

    results = []

    beam_methods = ["beam", "beam_lm", "beam_lm_rescore"]

    for beam_width in beam_widths:
        decoder = Wav2Vec2Decoder(lm_model_path=lm_model_path, beam_width=beam_width)

        for audio_path, target in test_samples:
            audio_input, sr = torchaudio.load(audio_path)
            sample_name = os.path.basename(audio_path)

            for method in beam_methods:
                transcript = decoder.decode(audio_input, method=method)

                cer = Levenshtein.distance(target, transcript) / len(target)
                wer = word_error_rate(target, transcript)

                results.append({
                    "Sample": sample_name,
                    "Method": method,
                    "Beam Width": beam_width,
                    "CER": cer,
                    "WER": wer
                })

    results_df = pd.DataFrame(results)

    summary = results_df.groupby(["Method", "Beam Width"])[["CER", "WER"]].mean().reset_index()

    print("\n===== BEAM WIDTH EXPERIMENT =====")
    print(summary)

    plt.figure(figsize=(12, 10))

    for i, method in enumerate(beam_methods):
        plt.subplot(3, 2, i * 2 + 1)
        method_data = summary[summary["Method"] == method]
        plt.plot(method_data["Beam Width"], method_data["CER"], marker='o')
        plt.title(f"{method} - CER vs Beam Width")
        plt.xlabel("Beam Width")
        plt.ylabel("CER")

        plt.subplot(3, 2, i * 2 + 2)
        plt.plot(method_data["Beam Width"], method_data["WER"], marker='o')
        plt.title(f"{method} - WER vs Beam Width")
        plt.xlabel("Beam Width")
        plt.ylabel("WER")

    plt.tight_layout()
    plt.savefig("beam_width_experiment.png")

    return results_df, summary


def experiment_alpha_beta(alphas=[0.5, 1.0, 1.5, 2.0],
                          betas=[0.0, 0.5, 1.0, 1.5],
                          lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz"):
    """
    Experiment with different alpha (LM weight) and beta (word bonus) values
    """
    test_samples = [
        ("examples/sample1.wav",
         "IF YOU ARE GENEROUS HERE IS A FITTING OPPORTUNITY FOR THE EXERCISE OF YOUR MAGNANIMITY IF YOU ARE PROUD HERE AM I YOUR RIVAL READY TO ACKNOWLEDGE MYSELF YOUR DEBTOR FOR AN ACT OF THE MOST NOBLE FORBEARANCE"),
        ("examples/sample3.wav",
         "GUESS A MAN GETS USED TO ANYTHING HELL MAYBE I CAN HIRE SOME BUMS TO SIT AROUND AND WHOOP IT UP WHEN THE SHIPS COME IN AND BILL THIS AS A REAL OLD MARTIAN DEN OF SIN"),
    ]

    results = []

    lm_methods = ["beam_lm", "beam_lm_rescore"]

    for alpha in alphas:
        for beta in betas:
            decoder = Wav2Vec2Decoder(lm_model_path=lm_model_path, alpha=alpha, beta=beta)

            for audio_path, target in test_samples:
                audio_input, sr = torchaudio.load(audio_path)
                sample_name = os.path.basename(audio_path)

                for method in lm_methods:
                    transcript = decoder.decode(audio_input, method=method)

                    cer = Levenshtein.distance(target, transcript) / len(target)
                    wer = word_error_rate(target, transcript)

                    results.append({
                        "Sample": sample_name,
                        "Method": method,
                        "Alpha": alpha,
                        "Beta": beta,
                        "CER": cer,
                        "WER": wer
                    })

    results_df = pd.DataFrame(results)

    summary = results_df.groupby(["Method", "Alpha", "Beta"])[["CER", "WER"]].mean().reset_index()

    print("\n===== ALPHA/BETA EXPERIMENT =====")
    print(summary)

    for method in lm_methods:
        method_data = summary[summary["Method"] == method].copy()

        cer_pivot = method_data.pivot_table(values="CER", index="Alpha", columns="Beta")
        wer_pivot = method_data.pivot_table(values="WER", index="Alpha", columns="Beta")

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(cer_pivot, cmap='viridis')
        plt.colorbar(label='CER')
        plt.title(f"{method} - CER by Alpha/Beta")
        plt.xlabel("Beta Index")
        plt.ylabel("Alpha Index")
        plt.xticks(range(len(betas)), [str(b) for b in betas])
        plt.yticks(range(len(alphas)), [str(a) for a in alphas])

        plt.subplot(1, 2, 2)
        plt.imshow(wer_pivot, cmap='viridis')
        plt.colorbar(label='WER')
        plt.title(f"{method} - WER by Alpha/Beta")
        plt.xlabel("Beta Index")
        plt.ylabel("Alpha Index")
        plt.xticks(range(len(betas)), [str(b) for b in betas])
        plt.yticks(range(len(alphas)), [str(a) for a in alphas])

        plt.tight_layout()
        plt.savefig(f"{method}_alpha_beta_experiment.png")

    return results_df, summary


def experiment_different_lm_models(lm_models=["lm/3-gram.pruned.1e-7.arpa.gz", "lm/4-gram.arpa.gz"]):
    """
    Experiment with different language models
    Note: You need to download and provide paths to the additional LM models
    """
    test_samples = [
        ("examples/sample1.wav",
         "IF YOU ARE GENEROUS HERE IS A FITTING OPPORTUNITY FOR THE EXERCISE OF YOUR MAGNANIMITY IF YOU ARE PROUD HERE AM I YOUR RIVAL READY TO ACKNOWLEDGE MYSELF YOUR DEBTOR FOR AN ACT OF THE MOST NOBLE FORBEARANCE"),
        ("examples/sample3.wav",
         "GUESS A MAN GETS USED TO ANYTHING HELL MAYBE I CAN HIRE SOME BUMS TO SIT AROUND AND WHOOP IT UP WHEN THE SHIPS COME IN AND BILL THIS AS A REAL OLD MARTIAN DEN OF SIN"),
        ("examples/sample6.wav", "AT THIS TIME ALL PARTICIPANTS ARE IN A LISTEN ONLY MODE"),
    ]

    results = []

    lm_methods = ["beam_lm", "beam_lm_rescore"]

    for lm_model in lm_models:
        lm_name = os.path.basename(lm_model).replace(".arpa.gz", "")

        try:
            decoder = Wav2Vec2Decoder(lm_model_path=lm_model)

            for audio_path, target in test_samples:
                audio_input, sr = torchaudio.load(audio_path)
                sample_name = os.path.basename(audio_path)

                for method in lm_methods:
                    transcript = decoder.decode(audio_input, method=method)

                    cer = Levenshtein.distance(target, transcript) / len(target)
                    wer = word_error_rate(target, transcript)

                    results.append({
                        "Sample": sample_name,
                        "Method": method,
                        "LM Model": lm_name,
                        "CER": cer,
                        "WER": wer
                    })
        except Exception as e:
            print(f"Error with LM model {lm_model}: {e}")

    if not results:
        print("No results collected. Make sure the LM models are available.")
        return None, None

    results_df = pd.DataFrame(results)

    summary = results_df.groupby(["Method", "LM Model"])[["CER", "WER"]].mean().reset_index()

    print("\n===== LM MODEL COMPARISON =====")
    print(summary)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for method in lm_methods:
        method_data = summary[summary["Method"] == method]
        plt.bar(
            [f"{model}_{method}" for model in method_data["LM Model"]],
            method_data["CER"]
        )
    plt.title("Character Error Rate by LM Model and Method")
    plt.ylabel("CER")
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    for method in lm_methods:
        method_data = summary[summary["Method"] == method]
        plt.bar(
            [f"{model}_{method}" for model in method_data["LM Model"]],
            method_data["WER"]
        )
    plt.title("Word Error Rate by LM Model and Method")
    plt.ylabel("WER")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("lm_models_comparison.png")

    return results_df, summary


def generate_report():
    """
    Run all experiments and generate a comprehensive report
    """
    print("=== STARTING EXPERIMENTS FOR ASR DECODING REPORT ===")

    # Experiment 1
    results_methods, summary_methods = run_decoding_experiments()

    # Experiment 2
    results_beam, summary_beam = experiment_beam_width()

    # Experiment 3
    results_alpha_beta, summary_alpha_beta = experiment_alpha_beta()

    # Experiment 4
    try:
        results_lm, summary_lm = experiment_different_lm_models()
        lm_experiment_ran = True
    except:
        print("Failed to run LM model comparison. Make sure alternative LM models are downloaded.")
        lm_experiment_ran = False

    with open("asr_decoding_report.md", "w") as f:
        f.write("# ASR Decoding Experiments Report\n\n")

        f.write("## 1. Comparison of Decoding Methods\n\n")
        f.write("We implemented and compared four ASR decoding methods:\n")
        f.write("- Greedy decoding\n")
        f.write("- Beam search decoding\n")
        f.write("- Beam search with LM scores fusion\n")
        f.write("- Beam search with a second pass LM rescoring\n\n")

        f.write("### Results\n\n")
        f.write(summary_methods.to_markdown(index=False))
        f.write("\n\n")
        f.write("![Decoding Methods Comparison](decoding_methods_comparison.png)\n\n")

        f.write("## 2. Effect of Beam Width\n\n")
        f.write("We experimented with different beam width values to observe the effect on decoding quality.\n\n")
        f.write("### Results\n\n")
        f.write(summary_beam.to_markdown(index=False))
        f.write("\n\n")
        f.write("![Beam Width Experiment](beam_width_experiment.png)\n\n")

        f.write("## 3. Effect of Alpha and Beta Parameters\n\n")
        f.write(
            "We varied the language model weight (alpha) and word insertion bonus (beta) to find optimal values.\n\n")
        f.write("### Results\n\n")
        f.write(summary_alpha_beta.to_markdown(index=False))
        f.write("\n\n")
        f.write("![Alpha/Beta Experiment for Beam LM](beam_lm_alpha_beta_experiment.png)\n\n")
        f.write("![Alpha/Beta Experiment for Beam LM Rescore](beam_lm_rescore_alpha_beta_experiment.png)\n\n")

        if lm_experiment_ran:
            f.write("## 4. Comparison of Different Language Models\n\n")
            f.write("We compared the performance of different n-gram language models.\n\n")
            f.write("### Results\n\n")
            f.write(summary_lm.to_markdown(index=False))
            f.write("\n\n")
            f.write("![LM Models Comparison](lm_models_comparison.png)\n\n")

        f.write("## Conclusion\n\n")
        f.write("Based on our experiments, we can draw the following conclusions:\n\n")

        best_method = summary_methods.loc[summary_methods["WER"].idxmin()]["Method"]
        best_wer = summary_methods.loc[summary_methods["WER"].idxmin()]["WER"]

        f.write(f"1. The best performing decoding method is **{best_method}** with a WER of {best_wer:.4f}.\n")

    print("\nReport generated: asr_decoding_report.md")


if __name__ == "__main__":
    generate_report()
