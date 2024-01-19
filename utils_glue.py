import torch
import torch.nn.functional as F

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

# kd loss
def cal_loss(s_logits, t_logits, temperature):
    soft_labels = F.log_softmax(t_logits / temperature, dim=-1, dtype=torch.float32)
    log_prob = F.log_softmax(s_logits / temperature, dim=-1, dtype=torch.float32)
    ori_kld_loss = -torch.exp(soft_labels) * log_prob + torch.exp(soft_labels) * soft_labels
    loss = torch.mean(torch.sum(ori_kld_loss, dim=-1))
    
    return loss

class LGTMTeacher(object):
    def __init__(self,teacher_model,student_model,alpha_kd,alpha_kd_t,optimizer_t,scheduler_t, temperature):
        self.temperature = temperature
        self.teacher_model = teacher_model
        self.student_model = student_model
        # for student
        self.alpha_kd = alpha_kd 
        # for teacher
        self.alpha_kd_t = alpha_kd_t 
        self.optimizer_t = optimizer_t
        self.scheduler_t = scheduler_t

    def cal_stu_tea_loss(self, teacher_outputs, student_outputs, flag=1):
        t_loss, t_logits = teacher_outputs.loss, teacher_outputs.logits
        loss, logits = student_outputs.loss, student_outputs.logits
        
        student_loss = None
        teacher_loss = None

        # if flag=0, calculate the student loss and teacher loss simultaneously
        if flag == 0:
            # for student
            t_soft_labels = t_logits.detach()
            s_kld_loss = cal_loss(logits, t_soft_labels, self.temperature)
            student_loss = self.alpha_kd * s_kld_loss + (1- self.alpha_kd) * loss
            
        # for teacher
        soft_labels = logits.detach()
        t_kld_loss = cal_loss(t_logits, soft_labels, self.temperature)
        teacher_loss = self.alpha_kd_t *  t_kld_loss + (1- self.alpha_kd_t) * t_loss
    
        return student_loss, teacher_loss
        
    def _hessian_vector_product(self, vector, input, r=1e-2):
        R = r / _concat(vector).norm() # episilon
        # vector is the gradients of the student's parameters on valuation set
        self.teacher_model.eval()
        self.student_model.eval()
        teacher_outputs = self.teacher_model(**input) # (loss), logits, (hidden_states), (attentions)

        for p, v in zip(self.student_model.parameters(), vector):
            p.data.add_(R, v)
        student_outputs = self.student_model(**input)
        _, loss_x = self.cal_stu_tea_loss(teacher_outputs, student_outputs)
        grads_p = torch.autograd.grad(loss_x, self.teacher_model.parameters())

        for p, v in zip(self.student_model.parameters(), vector):
            p.data.sub_(2 * R, v)
        teacher_outputs = self.teacher_model(**input) # (loss), logits, (hidden_states), (attentions)
        student_outputs = self.student_model(**input)
        _, loss_y = self.cal_stu_tea_loss(teacher_outputs, student_outputs)
        grads_n = torch.autograd.grad(loss_y, self.teacher_model.parameters())

        # recover the parameters of the student
        for p, v in zip(self.student_model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
