using LinearAlgebra
using NLPModels


function cauchy_grad(nlp :: AbstractNLPModel; tol = 1e-6, kmax = 10000, max_time = 30.0, m = 1e-4)

    f(x) = obj(nlp, x)
    x = copy(nlp.meta.x0) #copiamos (salvamos) o x0 (x inicial) para não alterá-lo
    g(x) = grad(nlp, x) #calculamos o gradiente de f em x_0
    t0 = time() #guardamos o tempo atual, será nosso tempo inicial
    variacao_t = 0.0  #guardaremos aqui o tempo que o algoritmo leva para encontrar a solução
    fx = f(x)
    t = 1.0
    k = 0 #contador de iterações
    gx = g(x) #guardamos na variável gx o valor do gradiente de f no ponto x (neste caso, gx = g(x_0))
    status = :unknown

    while !(norm(gx) < tol || k > kmax || variacao_t > max_time)  #enquanto não ultrapasse o número maximo de iterações/tempo; ||g(x)|| < tol
        d = - gx  #definimos a direção d = - gradiente
        xt = x + t * d #esse t é pela busca do dy
        gx = g(x)  #atualiza g(x) com o x encontrado acima
        ft = f(xt)
        while ft ≥ fx + m * t * dot(d, gx) #(BUSCA INEXATA DE ARMIJO)
            #Objetivo: encontrar t que minimize f(xk+t*d). Condição de parada do Armijo: xk entra na região de decréscimo
            t = 0.5*t #diminuímos t afim de que xt = xk+t*d caia dentro da região de decréscimo de f
            xt = x + t * d
            ft = f(xt) #guardamos em ft f(xk+t*d) para que o while seja verificado
            if t < 1e-16 #segunda condição de parada: tk se torna muito pequeno sem xk ter entrado na região de decréscimo, equivalente a "alcançou o máximo de iterações"
                status = :stalled
                break
            end
        end
        if status == :stalled
            break
        end
        x = xt
        fx = ft
        gx = g(x)
        k = k + 1  #atualiza o contador de passos
        variacao_t = time() - t0 # Tempo decorrido
    end

    if norm(gx) < tol
        status = :first_order
    elseif k > kmax
        status = :max_iter
    elseif variacao_t > max_time
        status = :max_time
    end

    return status, x, obj(nlp, x),norm(gx), k, variacao_t
end

function newton_modificado(nlp :: AbstractNLPModel; tol = 1e-6, kmax = 10000, max_time = 30.0, m = 1e-4)
    f(x) = obj(nlp, x)
    x = copy(nlp.meta.x0) #copiando o x0 para não alterar seus valores, usaremos ele em outros algoritmos
    g(x) = grad(nlp, x) #calculando o gradiente de f em x0
    H(x) = Symmetric(hess(nlp, x), :L)
    t0 = time() #guardamos o tempo atual, será nosso tempo inicial
    variacao_t = 0.0 #guardaremos aqui o tempo que o algoritmo leva para encontrar a solução
    fx = f(x) #guardamos em fx o valor de f no ponto x0

    k = 0 #contador de iterações
    gx = g(x) #guardamos na variável gx o valor do gradiente de f no ponto x (em específico, nesse caso, g(x0))
    status = :unknown
    while !(norm(gx) < tol || k > kmax || variacao_t > max_time) #transferindo para a teoria: norm(gx) < tol significa g(x) = 0
        d = -H(x) \ gx #resolvemos o sistema linear -(Hessiana da f)d = (Gradiente da f) e guardamos d, uma suposta direção de descida
        if dot(d, gx) >= 0 #caso em que d não é direção de descida (isso acontece quando a Hessiana da f não é definida positiva)
            d = -d
        end
        t = 1.0 #t inicial (t0) que usaremos na busca inexata (Armijo)
        xt = x + d #nosso xk + t * d inicial (t = 1)
        ft = f(xt) #guardamos na variável ft o valor de f(x+t*d) para t = 1
        while ft ≥ fx + m * t * dot(d, gx) #(BUSCA INEXATA DE ARMIJO)
            #Objetivo: encontrar t que minimize f(xk+t*d). Condição de parada do Armijo: xk entra na região de decréscimo
            t = 0.5*t #diminuímos t afim de que xt = xk+t*d caia dentro da região de decréscimo de f
            xt = x + t * d
            ft = f(xt) #guardamos em ft f(xk+t*d) para que o while seja verificado
            if t < 1e-16 #segunda condição de parada: tk se torna muito pequeno sem xk ter entrado na região de decréscimo, equivalente a "alcançou o máximo de iterações"
                status = :stalled
                break
            end
        end
        if status == :stalled
            break
        end
        x = xt #atualizamos x(k+1) como sendo o xk resultado da busca de Armijo
        fx = ft #precisaremos da f(xk(+1)) para a Busca de Armijo da próxima iteração (caso haja)
        gx = g(x) #guardamos em gx o Gradiente de f no ponto x(k+1) para ver se a condição do While foi satisfeita
        k = k + 1
        variacao_t = time() - t0 #anotamos o tempo decorrido para verificar se o tempo máximo do algoritmo foi atingido
    end

    if norm(gx) < tol
        status = :first_order
    elseif k > kmax
        status = :max_iter
    elseif variacao_t > max_time
        status = :max_time
    end

   return status, x, obj(nlp, x),norm(gx), k, variacao_t
end

function newton_puro(nlp :: AbstractNLPModel; tol = 1e-6, kmax = 10000, max_time = 30.0) #marina = meewton puro não tem busca

    f(x) = obj(nlp, x)
    x = copy(nlp.meta.x0) #copiamos (salvamos) o x0 (x inicial) para não alterá-lo
    g(x) = grad(nlp, x) #calculamos o gradiente de f em x_0
    H(x) = Symmetric(hess(nlp, x), :L) #
    t0 = time() #guardamos o tempo atual, será nosso tempo inicial
    variacao_t = 0.0  #guardaremos aqui o tempo que o algoritmo leva para encontrar a solução

    k = 0 #contador de iterações
    gx = g(x) #guardamos na variável gx o valor do gradiente de f no ponto x (neste caso, gx = g(x_0))
    while !(norm(gx) < tol || k > kmax || variacao_t > max_time) #enquanto não ultrapasse o número maximo de iterações/tempo; ||g(x)|| < tol
        d = -H(x) \ gx #define a direção de descida resolvendo o sistema Hd = δgx
        x = x + d #atualiza o x
        gx = g(x) #atualiza g(x) com o x encontrado acima
        k = k + 1 #atualiza o contador de passos
        variacao_t = time() - t0 # Tempo decorrido
    end

    status = :unknown

    if norm(gx) < tol
        status = :first_order
    elseif k > kmax
        status = :max_iter
    elseif variacao_t > max_time
        status = :max_time
    end

    return status, x, obj(nlp, x), norm(gx), k,  variacao_t
end

function quase_newton_BFGS(nlp :: AbstractNLPModel; tol = 1e-6, kmax = 10000, max_time = 30.0, m = 1e-4)

    f(x) = obj(nlp, x)
    x = copy(nlp.meta.x0) #copiando o x0 para não alterar seus valores, usaremos ele em outros algoritmos
    g(x) = grad(nlp, x) #calculando o gradiente de f em x0
    H(x) = Symmetric(hess(nlp, x), :L)
    t0 = time() #guardamos o tempo atual, será nosso tempo inicial
    variacao_t = 0.0 #guardaremos aqui o tempo que o algoritmo leva para encontrar a solução
    fx = f(x)

    k = 0 #contador de iterações
    Bk = zeros(length(x),length(x))+I #Precisamos começar com uma matriz B0 simétrica e definida positiva, portando, pegamos B0 = Matriz Identidade.
    #Com isso, teremos que o primeiro passo do algoritmo de Quase Newton será análogo ao de Cauchy.
    #Entretanto, ao atualizarmos a função Bk, teremos aproximações diferentes da Identidade para a Hessiana, e com isso, os passos restantes
    #do algoritmo se comportarão diferentes do método de Cauchy
    gx = g(x) #guardamos na variável gx o valor do gradiente de f no ponto x (em específico, nesse caso, g(x0))
    status = :unknown
    while !(norm(gx) < tol || k > kmax || variacao_t > max_time) #transferindo para a teoria: norm(gx) < tol significa g(x) = 0
        d = -Bk \ gx #resolvemos o sistema linear -(Hessiana da f)d = (Gradiente da f) e guardamos d, uma suposta direção de descida
        if dot(d, gx) >= 0 #caso em que d não é direção de descida (isso acontece quando a Hessiana da f não é definida positiva) ?? PQ
            d = -d
        end
        t = 1.0 #t inicial (t0) que usaremos na busca inexata (Armijo)
        xt = x + d #nosso xk + t * d inicial (t = 1)
        ft = f(xt) #guardamos na variável ft o valor de f(x+t*d) para t = 1
        while ft ≥ fx + m * t * dot(d, gx) #(BUSCA INEXATA DE ARMIJO)
            #Objetivo: encontrar t que minimize f(xk+t*d). Condição de parada do Armijo: xk entra na região de decréscimo
            t = 0.5t #diminuímos t afim de que xt = xk+t*d caia dentro da região de decréscimo de f
            xt = x + t * d
            ft = f(xt) #guardamos em ft f(xk+t*d) para que o while seja verificado
            if t < 1e-16 #segunda condição de parada: tk se torna muito pequeno sem xk ter entrado na região de decréscimo, equivalente a "alcançou o máximo de iterações"
                status = :stalled
                break
            end
        end
        if status == :stalled
            break
        end
        Sk = xt - x #utilizaremos esse termo para atualizar a Bk
        yk = g(xt) - gx #utilizaremos esse termo para atualizar a Bk
        Termo1 = Bk*Sk #Testar usar gx = Bk*Sk para economizar uma alocação (otimização) ?????????????????????
        Bk = Bk - (Termo1*Termo1')/dot(Sk,Termo1) + (yk*yk')/dot(yk,Sk) #Atualizando Bk
        x = xt #atualizamos x(k+1) como sendo o xk resultado da busca de Armijo
        fx = ft #precisaremos da f(xk(+1)) para a Busca de Armijo da próxima iteração e para definir a B(k+1) (caso haja)
        gx = g(x) #guardamos em gx o Gradiente de f no ponto x(k+1) para ver se a condição do While foi satisfeita
        k = k + 1
        variacao_t = time() - t0 #anotamos o tempo decorrido para verificar se o tempo máximo do algoritmo foi atingido
    end

    if norm(gx) < tol
        status = :first_order
    elseif k > kmax
        status = :max_iter
    elseif variacao_t > max_time
        status = :max_time
    end

    return  status, x, obj(nlp, x),norm(gx), k, variacao_t
end

function lagrangeano(nlp_i, metodo;
                            tol_g = 1e-6, tol_h = 1e-6, kmax = 1000, max_time = 30.0)

    f, x, λ, ρ, h = nlp_i #dados iniciais
    t0 = time()
    variacao_t = 0.0
    status = :unknown
    k = 0
    L(x) = f(x) + λ'*h(x) + (ρ/2)*norm(h(x))^2 #função L(x,lambda,ro) -> lagrangiano aumentado
    nlp = ADNLPModel(L, x)
    g(x) = grad(nlp, x)
    ngx = norm(g(x))
    nhx = norm(h(x))
    while (ngx>tol_g || nhx>tol_h) && k<=kmax && variacao_t<=max_time #Ou seja, enquanto as condições de KKT não forem satisfeitas:
        status_met, x_k, obj_met, gx_met, k_met, variacao_tmet = metodo(nlp) #Pegamos as informações do método que utilizamos para resolver o subproblema
        if status_met != :first_order #Se o status gerado pelo método não for de sucesso (first_order), então:
            status = :stalled_subproblem #colocamos o status como "parado no subproblema" e quebramos o código
            variacao_t = time() - t0
            break
        end
        λ = λ + ρ * h(x_k) #atualizando o multiplicador de Lagrange
        if norm(h(x_k)) >= 0.1 * norm(h(x))
            ρ = 2*ρ #atualizando o parâmetro de penalidade caso o algoritmo não tenha progredido muito em sua última iteração
        end
        x = x_k
        L(x) = f(x) + λ'*h(x) + (ρ/2)*norm(h(x))^2 # Nova função Lagrangiano aumentado a ser minimizada
        nlp = ADNLPModel(L, x) #atualizando variáveis
        g(x) = grad(nlp, x)
        ngx = norm(g(x))
        nhx = norm(h(x))
        k = k + 1
        variacao_t = time() - t0
    end

    if ngx < tol_g && nhx < tol_h
        status =:first_order
    elseif k > kmax
        status = :max_iter
    elseif variacao_t > max_time
        status = :max_time
    end
     return status, x, variacao_t, k, ngx, nhx
end

metodos = [newton_puro, newton_modificado, cauchy_grad, quase_newton_BFGS]
problemas = [ #a      b     p      λ     ρ        x₀
             [ [2,1], 8, [10,7], [1.0], 1.0, [-1.2; 1.0] ],
             [ [-3/5,2], 8, [10,7], [1.0], 1.0, [-1.2; 1.0]],
             [ [5,5,5,5,5,5,5,5,5,5], 10, [1,2,3,4,5,6,7,8,9,10], [1.0], 1.0, [-1.2; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0]]]

for metodo in metodos
    println("MÉTODO: $metodo")
    for prob in problemas
        a, b, p, λ, ρ, x₀ = prob
        f(x) = 1/2*norm(x - p)^2
        h(x) = [a'x - b]
        nlp_i = [f, x₀, λ, ρ, h]
        #################
        status, x, variacao_t, k, ngx, nhx = lagrangeano(nlp_i, metodo)
        println("status: ",status)
        println("x: ",x)
        println("norma(grad): ",ngx)
        println("norma(h): ", nhx)
        println("N de iterações: ",k)
        println("tempo: ", variacao_t)
        print("\n\n")
        #################
    end
end
