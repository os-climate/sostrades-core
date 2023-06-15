'''
Copyright 2022 Airbus SAS

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
load/dump - read/write feature to manage load and dump of exported study data
'''
from pickle import UnpicklingError, dumps as pkl_dumps, loads as pkl_loads
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
from Crypto.Cipher import AES
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Protocol.KDF import PBKDF2
import pandas as pd


class LoadDumpException(Exception):
    def __init__(self, encryption_mode, action):
        super().__init__(
            f'Error when {action} with mode {encryption_mode}. ABORTED')


class AbstractLoadDump:

    def load(self, f_name):
        raise NotImplementedError()

    def dump(self, dict_obj, f_name):
        raise NotImplementedError()


class DirectLoadDump(AbstractLoadDump):

    def load(self, f_name):
        with open(f_name, 'rb') as d_s:
            try:
                loaded_dict = pd.read_pickle(d_s)
            except UnpicklingError:
                raise LoadDumpException('direct', 'loading')
            except UnicodeDecodeError:
                raise LoadDumpException('direct', 'loading')
            except ValueError:
                raise LoadDumpException('direct', 'loading')
            except:
                raise
        return loaded_dict

    def dump(self, dict_obj, f_name):
        with open(f_name, 'wb') as d_s:
            pd.to_pickle(dict_obj, d_s)


class CryptedLoadDump(AbstractLoadDump):
    '''
    Encryption feature to securise load and dump of exported study data
    '''
    key_enc_basename = 'key.bin.enc'

    def __init__(self, private_key_file, public_key_file):
        self.private_key_file = private_key_file
        self.public_key_file = public_key_file

    def load(self, f_name):
        bytes_obj = self.decrypt_file(f_name)
        # decode the bytes object to data dict
        loaded_dict = pkl_loads(bytes_obj)
        return loaded_dict

    def dump(self, dict_obj, f_name):
        # Hear cannot use panda pickelization method because it does not work
        # with object
        bytes_obj = pkl_dumps(dict_obj)
        self.encrypt_stream(bytes_obj, f_name)

    def get_key_enc_file(self, enc_filepath):
        return str(enc_filepath) + '.' + self.key_enc_basename

    def encrypt_stream(self, bytes_to_en, encrypted_file):
        # Generate random key
        # 32 bytes * 8 = 256 bits (1 byte = 8 bits)
        iv = get_random_bytes(16)
        key = PBKDF2(iv, b'', dkLen=32)
        # To modify path to public key
        with open(self.public_key_file, 'r') as p_k_s:
            public_key_string = p_k_s.read()
        public_key = RSA.importKey(public_key_string)
        cipher = PKCS1_OAEP.new(public_key)
        # Encrypt random key
        encrypted_secret_key = cipher.encrypt(key)
        # Write it out, to modify with path where to store the random key
        key_enc_f = self.get_key_enc_file(encrypted_file)
        with open(key_enc_f, 'wb') as output_key_s:
            output_key_s.write(encrypted_secret_key)
        # Encrypt stream with cipher created from random key
        cipher = AES.new(key, AES.MODE_CBC, iv)
        ciphered_data = cipher.encrypt(pad(bytes_to_en, AES.block_size))

        # Write encrypted stream inside a file
        with open(encrypted_file, 'wb') as file_out_s:
            file_out_s.write(iv)
            file_out_s.write(ciphered_data)

    def decrypt_file(self, in_file):
        # To modify with path where random key is stored
        key_enc_f = self.get_key_enc_file(in_file)
        try:
            with open(key_enc_f, 'rb') as e_k_s:
                enc_key = e_k_s.read()
        except FileNotFoundError:
            raise LoadDumpException('encryption', 'loading/decrypting')
        # To modify path to private key
        with open(self.private_key_file, 'r') as p_k_s:
            private_key_string = p_k_s.read()
        private_key = RSA.importKey(private_key_string)
        cipher = PKCS1_OAEP.new(private_key)
        # Decrypt random key with RSA private key
        key = cipher.decrypt(enc_key)

        # Read the data from the file
        with open(in_file, 'rb') as enc_f_s:
            iv = enc_f_s.read(16)  # Read the iv out - this is 16 bytes long
            ciphered_data = enc_f_s.read()  # Read the rest of the data

        cipher = AES.new(key, AES.MODE_CBC, iv)  # Setup cipher
        # Decrypt and then up-pad the result
        original_data = unpad(cipher.decrypt(ciphered_data), AES.block_size)

        return original_data
