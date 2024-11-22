import torch.nn as nn
import torch.nn.functional as F
import torch


class ChannelAttentionModule3D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModule3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # Fully connected layers for reduction and expansion
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, d, h, w = x.size()
        
        # Average pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1, 1)
        
        # Max pooling
        max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1, 1)
        
        # Combine and apply sigmoid
        out = self.sigmoid(avg_out + max_out)
        
        return x * out  # Scale input by channel attention weights


class SpatialAttentionModule3D(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule3D, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global max pooling along the channel axis
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Global average pooling along the channel axis
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        # Concatenate max and avg along channel dimension
        combined = torch.cat([max_out, avg_out], dim=1)
        
        # Apply convolution and sigmoid
        attention = self.sigmoid(self.conv(combined))
        
        return x * attention  # Scale input by spatial attention weights


class CBAM3D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM3D, self).__init__()
        self.channel_attention = ChannelAttentionModule3D(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttentionModule3D()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        
        return x

class conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):
        """
        + Instantiate modules: conv-elu-norm
        + Assign them as member variables
        """
        super(conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.elu = nn.ELU(inplace=True)
        # with learnable parameters
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        # self.dropout = nn.Dropout3d(dropout_prob) 


    def forward(self, x):
        return self.elu(self.norm(self.conv(x)))


class conv3d_x3(nn.Module):
    """Three serial convs with a residual connection.
    Structure:
        inputs --> ① --> ② --> ③ --> outputs
                   ↓ --> add--> ↑
    """

    def __init__(self, in_channels, out_channels):
        super(conv3d_x3, self).__init__()
        self.conv_1 = conv3d(in_channels, out_channels)
        self.conv_2 = conv3d(out_channels, out_channels)
        self.conv_3 = conv3d(out_channels, out_channels)
        self.skip_connection=nn.Conv3d(in_channels,out_channels,1)

    def forward(self, x):
        z_1 = self.conv_1(x)
        z_3 = self.conv_3(self.conv_2(z_1))
        return z_3 + self.skip_connection(x)

class conv3d_x2(nn.Module):
    """Three serial convs with a residual connection.
    Structure:
        inputs --> ① --> ② --> ③ --> outputs
                   ↓ --> add--> ↑
    """

    def __init__(self, in_channels, out_channels):
        super(conv3d_x2, self).__init__()
        self.conv_1 = conv3d(in_channels, out_channels)
        self.conv_2 = conv3d(out_channels, out_channels)
        self.skip_connection=nn.Conv3d(in_channels,out_channels,1)

    def forward(self, x):
        z_1 = self.conv_1(x)
        z_2 = self.conv_2(z_1)
        return z_2 + self.skip_connection(x)


class conv3d_x1(nn.Module):
    """Three serial convs with a residual connection.
    Structure:
        inputs --> ① --> ② --> ③ --> outputs
                   ↓ --> add--> ↑
    """

    def __init__(self, in_channels, out_channels):
        super(conv3d_x1, self).__init__()
        self.conv_1 = conv3d(in_channels, out_channels)
        self.skip_connection=nn.Conv3d(in_channels,out_channels,1)

    def forward(self, x):
        z_1 = self.conv_1(x)
        return z_1 + self.skip_connection(x)

class deconv3d_x3(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):
        super(deconv3d_x3, self).__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, 2, 2)
        self.lhs_conv = conv3d(out_channels // 2, out_channels)
        self.conv_x3 = nn.Sequential(
            nn.Conv3d(2*out_channels, out_channels,5,1,2),
            nn.ELU(inplace=True),
            # nn.Dropout3d(dropout_prob),
            nn.Conv3d(out_channels, out_channels,5,1,2),
            nn.ELU(inplace=True),
            # nn.Dropout3d(dropout_prob),
            nn.Conv3d(out_channels, out_channels,5,1,2),
            nn.ELU(inplace=True),
            # nn.Dropout3d(dropout_prob)
        )

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)
        lhs_conv = self.lhs_conv(lhs)
        rhs_add = torch.cat((rhs_up, lhs_conv),dim=1) 
        return self.conv_x3(rhs_add)+ rhs_up

class deconv3d_x2(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):
        super(deconv3d_x2, self).__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, 2, 2)
        self.lhs_conv = conv3d(out_channels // 2, out_channels)
        self.conv_x2= nn.Sequential(
            nn.Conv3d(2*out_channels, out_channels,5,1,2),
            nn.ELU(inplace=True),
            # nn.Dropout3d(dropout_prob),
            nn.Conv3d(out_channels, out_channels,5,1,2),
            nn.ELU(inplace=True),
            # nn.Dropout3d(dropout_prob),
        )

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)
        lhs_conv = self.lhs_conv(lhs)
        rhs_add = torch.cat((rhs_up, lhs_conv),dim=1) 
        return self.conv_x2(rhs_add)+ rhs_up

class deconv3d_x1(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):
        super(deconv3d_x1, self).__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, 2, 2)
        self.lhs_conv = conv3d(out_channels // 2, out_channels)
        self.conv_x1 = nn.Sequential(
            nn.Conv3d(2*out_channels, out_channels,5,1,2),
            nn.ELU(inplace=True),
            # nn.Dropout3d(dropout_prob)
        )

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)
        lhs_conv = self.lhs_conv(lhs)
        rhs_add = torch.cat((rhs_up, lhs_conv),dim=1) 
        return self.conv_x1(rhs_add)+ rhs_up 
        # Martin: vuelve a sumar el rhs_up  
        # para que sus caracteristicas ampliadas no se pierdan. Pero ya lo has hecho al
        # concatenar y pasar una convolucion... Tan importantes son las caracteristicas que vienen
        # ampliadas? Osea son mas importantes que las de la imagen original, pero como de importantes?
        # como se que son mas importantes? segun imagen? 
        

def conv3d_as_pool(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=0),
        nn.ELU(inplace=True))


def deconv3d_as_up(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride),
        nn.ELU(inplace=True)
    )


class softmax_out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(softmax_out, self).__init__()
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0)
        """
            Martin: no entiendo el hecho de hacer una concolucion de 1x1x1 
            cuando el num de canales ya es 1... Pero, ¿por que?:
            torch.equal(self.conv_1(x), self.conv_2(self.conv_1(x))) == False 
            si no cambia los canales que le esta haciendo a los valores? 
        """
        
    def forward(self, x):
        """Output with shape [batch_size, 1, depth, height, width]."""
        # Do NOT add normalize layer, or its values vanish.
        y_conv = self.conv_2(self.conv_1(x))
        
        """ Hacer return de dos canales hace que luego tengamos que usar funciones no diferenciables
        , por lo que el grafo de gradiantes se rompe... Mejor usar una softmax o en este caso
        un sigmoide... Sigmoid es lo mismo que softmax pero solamente para una clase. """
        # # y_conv = y_conv.permute(0, 2, 3, 4, 1).contiguous()
        # # y_conv = y_conv.view(y_conv.size(0), y_conv.numel() // (2 * y_conv.size(0)), 2)
        # y_conv = F.softmax(y_conv,dim=1)
        # return y_conv
    
        return nn.Sigmoid()(y_conv).squeeze(1)
        """
            Martin: Sigmoid solo saca un mapa de probabilidades que se supone que es la probilidad
            en este caso de tener cancer. Pero la red de entrenamiento espera encontrar dos mapas
            de probabilidades y eleigir la que tenga mas probabilidades en el mismo indice. 
            
            input = torch.tensor([
                [[0.8, 0.4, 0.3],  # Clase 0 probabilidades
                [0.1, 0.7, 0.2],  # Clase 0 probabilidades
                [0.6, 0.5, 0.1]],

                [[0.2, 0.6, 0.7],  # Clase 1 probabilidades
                [0.9, 0.3, 0.8],  # Clase 1 probabilidades
                [0.4, 0.5, 0.9]]  # Clase 1 probabilidades
            ])
            
            _, result_ = input.max(0) # funcion no complementaria, rompe el grafo de compilacion, pero que da igual al final uso softdice
            
            result_ = tensor([
                [0, 1, 1],  
                [1, 0, 1], 
                [0, 0, 1]
            ])
            
            Para conseguir esto necesitamos 
        """


class VNet_CBAM(nn.Module):
    def __init__(self, dig_sep):
        super(VNet_CBAM, self).__init__()
        # Capa inicial
        self.conv_1 = conv3d_x1(dig_sep, 16)
        self.pool_1 = conv3d_as_pool(16, 32)
        self.cbam_1 = CBAM3D(16)  

        # Segunda capa
        self.conv_2 = conv3d_x2(32, 32)
        self.pool_2 = conv3d_as_pool(32, 64)
        self.cbam_2 = CBAM3D(32)  

        # Tercera capa
        self.conv_3 = conv3d_x3(64, 64)
        self.pool_3 = conv3d_as_pool(64, 128)
        self.cbam_3 = CBAM3D(64)  

        # Cuarta capa
        self.conv_4 = conv3d_x3(128, 128)
        self.pool_4 = conv3d_as_pool(128, 256)
        self.cbam_4 = CBAM3D(128)  

        # Fondo
        self.bottom = conv3d_x3(256, 256)

        # Decodificador
        self.deconv_4 = deconv3d_x3(256, 256)
        self.deconv_3 = deconv3d_x3(256, 128)
        self.deconv_2 = deconv3d_x2(128, 64)
        self.deconv_1 = deconv3d_x1(64, 32)

        # Salida
        self.out = softmax_out(32, 1)

    def forward(self, x):
        # Encoder con CBAM y pooling
        conv_1 = self.conv_1(x)
        attention_1 = self.cbam_1(conv_1)
        pool_1 = self.pool_1(attention_1)

        conv_2 = self.conv_2(pool_1)
        attention_2 = self.cbam_2(conv_2)
        pool_2 = self.pool_2(attention_2)

        conv_3 = self.conv_3(pool_2)
        attention_3 = self.cbam_3(conv_3)
        pool_3 = self.pool_3(attention_3)

        conv_4 = self.conv_4(pool_3)
        attention_4 = self.cbam_4(conv_4)
        pool_4 = self.pool_4(attention_4)

        # Fondo
        bottom = self.bottom(pool_4)

        # Decoder utilizando skip connections con CBAM
        deconv_4 = self.deconv_4(attention_4, bottom)
        deconv_3 = self.deconv_3(attention_3, deconv_4)
        deconv_2 = self.deconv_2(attention_2, deconv_3)
        deconv_1 = self.deconv_1(attention_1, deconv_2)

        # Salida final
        return self.out(deconv_1)