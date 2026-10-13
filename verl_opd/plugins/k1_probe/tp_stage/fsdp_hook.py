    if config.distillation_loss.loss_mode == "k1_transplant":
        out.update(compute_transplant(student_logits, data))
